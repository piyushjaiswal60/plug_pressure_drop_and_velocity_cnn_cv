"""
ml_model/02_train_model.py   (v2 - single, consistent model-selection protocol)

Predicts plug-flow pressure drop (Delta P) with 6 models:
    Ridge, SVR (RBF), Random Forest, Gradient Boosting, XGBoost, AdaBoost
plus a "predict the mean" baseline.

WHY THE OLD SCRIPT GAVE TWO DIFFERENT WINNERS
---------------------------------------------
The old script ranked models twice:
  (a) grid-search score on ALL data  -> XGBoost looked best
  (b) nested CV                      -> Gradient Boosting looked best
(a) is optimistic: the best of hundreds of configurations is picked using the
very same samples that score it, and it used a different CV scheme / metric
definition than (b). They can never be compared fairly.

THE PROTOCOL USED HERE (one ranking, no contradictions)
-------------------------------------------------------
 1. NESTED CV decides which MODEL is best.
      outer loop : Repeated K-Fold   -> unbiased performance estimate (+/- std)
      inner loop : Repeated K-Fold GridSearchCV -> tunes hyper-parameters using
                   only the training part of each outer fold.
    The ranking is based ONLY on this nested-CV RMSE.
 2. A paired Wilcoxon test on per-sample errors tells you whether the winner is
    really better than each other model or whether they are statistically tied.
 3. The winning model is re-tuned ONCE on all data with the SAME inner CV and
    SAME metric; those hyper-parameters are the ones saved.
    The reported expected error of the saved model is its nested-CV RMSE
    (never its tuning score).
 4. Everything is written to results_dir/, and the best model is saved to
    models/ml/pressure_model.pkl (+ ml_model/pressure_model.pkl), so
    03_predict.py keeps working unchanged.

Outputs (test_results/):
    model_selection_report.xlsx  ranking, tuned params of every model, stability
    nested_cv_summary.xlsx       mean +/- std of every metric per model
    nested_cv_fold_details.xlsx  params chosen in every outer fold
    gridsearch_<model>.xlsx      full tuning results of each final grid search
    final_model_summary.xlsx     tuned params + nested-CV score per model
    best_model_info.json         name, params, expected error of the saved model
    prediction_plot.jpg / model_comparison.jpg / feature_importance.jpg

Usage:
    python 02_train_model.py                      # full run
    python 02_train_model.py --quick              # fast smoke test (small grids)
    python 02_train_model.py --input my_data.xlsx
"""

import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
SEED = 42
TARGET = "Delta P (Pa)"

OUTER_SPLITS = 5        # outer K-fold (performance estimate)
OUTER_REPEATS = 5       # repeats of the outer loop -> gives a real +/- std
INNER_SPLITS = 5        # inner K-fold (tuning)
INNER_REPEATS = 2       # repeats of the inner loop (less noisy tuning)
FINAL_SPLITS = 5        # final tuning on all data: SAME scheme as the inner loop
FINAL_REPEATS = 2

# One metric everywhere. MSE averages correctly over folds of unequal size;
# we always report its square root (RMSE).
GRID_SCORING = "neg_mean_squared_error"
N_JOBS = -1
ALPHA = 0.05            # significance level for the paired test


# --------------------------------------------------------------------------- #
# MODELS AND HYPER-PARAMETER GRIDS
# (compact on purpose: with ~20 samples, huge grids mostly fit noise)
# --------------------------------------------------------------------------- #
def build_models(quick=False):
    models = {}

    models["Ridge"] = (
        Ridge(),
        {"regressor__alpha": [0.01, 0.1, 1, 10, 100] if not quick else [0.1, 1, 10]},
    )

    # SVR is scale sensitive in the target too -> standardise y inside the model
    models["SVR (RBF)"] = (
        TransformedTargetRegressor(regressor=SVR(kernel="rbf"), transformer=StandardScaler()),
        {
            "regressor__regressor__C": [0.3, 1, 3, 10, 30] if not quick else [1, 10],
            "regressor__regressor__epsilon": [0.05, 0.1, 0.2] if not quick else [0.1],
            "regressor__regressor__gamma": ["scale", 0.01, 0.03, 0.1] if not quick else ["scale", 0.03],
        },
    )

    models["Random Forest"] = (
        RandomForestRegressor(random_state=SEED, n_jobs=1),
        {
            "regressor__n_estimators": [200, 500] if not quick else [100],
            "regressor__max_depth": [2, 3, 5, None] if not quick else [3, 5],
            "regressor__min_samples_leaf": [1, 2, 3] if not quick else [1, 2],
            "regressor__max_features": [1.0, 0.5] if not quick else [1.0],
        },
    )

    models["Gradient Boosting"] = (
        GradientBoostingRegressor(random_state=SEED),
        {
            "regressor__n_estimators": [50, 100, 200] if not quick else [50, 100],
            "regressor__learning_rate": [0.03, 0.1] if not quick else [0.1],
            "regressor__max_depth": [2, 3] if not quick else [2],
            "regressor__subsample": [0.7, 1.0] if not quick else [0.7],
        },
    )

    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(objective="reg:squarederror", tree_method="hist",
                         random_state=SEED, n_jobs=1, verbosity=0),
            {
                "regressor__n_estimators": [100, 200] if not quick else [100],
                "regressor__max_depth": [2, 3] if not quick else [2],
                "regressor__learning_rate": [0.03, 0.1] if not quick else [0.1],
                "regressor__subsample": [0.7, 1.0] if not quick else [0.7],
                "regressor__colsample_bytree": [0.7, 1.0] if not quick else [1.0],
                "regressor__min_child_weight": [1, 3] if not quick else [1],
            },
        )
    else:
        print("WARNING: xgboost not installed (pip install xgboost) -> XGBoost skipped.")

    # scikit-learn >= 1.2 uses `estimator`, older versions `base_estimator`
    try:
        ada = AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=SEED)
        depth_key = "regressor__estimator__max_depth"
    except TypeError:
        ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), random_state=SEED)
        depth_key = "regressor__base_estimator__max_depth"
    models["AdaBoost"] = (
        ada,
        {
            "regressor__n_estimators": [50, 100, 200] if not quick else [50],
            "regressor__learning_rate": [0.1, 0.5, 1.0] if not quick else [0.5],
            "regressor__loss": ["linear", "square", "exponential"] if not quick else ["linear"],
            depth_key: [2, 3] if not quick else [2],
        },
    )
    return models


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def make_pipeline(regressor, numeric_cols, categorical_cols):
    """Preprocessing + regressor in one object (required by 03_predict.py)."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    return Pipeline([("preprocessor", pre), ("regressor", regressor)])


def clean_params(params):
    return {k.replace("regressor__", ""): v for k, v in params.items()}


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def nested_cv(name, pipeline, grid, X, y):
    """Repeated nested CV.
    Returns: per-repeat metrics, list of per-repeat OOF predictions, fold table."""
    n = len(X)
    outer_k = min(OUTER_SPLITS, n)
    repeat_metrics, oof_list, fold_rows = [], [], []

    for rep in range(OUTER_REPEATS):
        outer = KFold(n_splits=outer_k, shuffle=True, random_state=SEED + rep)
        oof = np.zeros(n)
        for fold, (tr, te) in enumerate(outer.split(X)):
            if grid:
                inner = RepeatedKFold(n_splits=min(INNER_SPLITS, len(tr)),
                                      n_repeats=INNER_REPEATS, random_state=SEED)
                gs = GridSearchCV(pipeline, grid, cv=inner, scoring=GRID_SCORING,
                                  n_jobs=N_JOBS, refit=True)
                gs.fit(X.iloc[tr], y.iloc[tr])
                oof[te] = gs.predict(X.iloc[te])
                fold_rows.append({
                    "Model": name, "Repeat": rep + 1, "Outer fold": fold + 1,
                    "Inner best RMSE": float(np.sqrt(-gs.best_score_)),
                    **clean_params(gs.best_params_),
                })
            else:
                est = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
                oof[te] = est.predict(X.iloc[te])
        repeat_metrics.append(regression_metrics(y.values, oof))
        oof_list.append(oof)
    return pd.DataFrame(repeat_metrics), oof_list, pd.DataFrame(fold_rows)


def param_stability(fold_df):
    """How often did the outer folds agree on each hyper-parameter value?"""
    rows = []
    skip = {"Model", "Repeat", "Outer fold", "Inner best RMSE"}
    for model, g in fold_df.groupby("Model"):
        for col in g.columns:
            if col in skip or g[col].isna().all():
                continue
            vc = g[col].astype(str).value_counts(normalize=True)
            rows.append({"Model": model, "Hyper-parameter": col,
                         "Most chosen value": vc.index[0],
                         "Chosen in % of outer folds": round(100 * vc.iloc[0], 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def train_pressure_model(
    input_file="data/ml_data/cleaned_data.xlsx",
    model_file="models/ml/pressure_model.pkl",
    results_dir="test_results",
    quick=False,
):
    global OUTER_REPEATS, INNER_REPEATS, FINAL_REPEATS
    if quick:
        OUTER_REPEATS, INNER_REPEATS, FINAL_REPEATS = 2, 1, 1
        print(">>> QUICK MODE: tiny grids / few repeats - results are NOT final <<<")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_file) or ".", exist_ok=True)
    os.makedirs("ml_model", exist_ok=True)

    print(f"Loading cleaned data from {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    if df.empty or TARGET not in df.columns:
        print("Error: dataset empty or target column missing.")
        return

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Samples: {len(df)} | Categorical: {categorical_cols} | Numeric: {numeric_cols}")
    if len(df) < 30:
        print("NOTE: very small dataset -> differences between models are usually within "
              "noise. Read the paired-test column before trusting the ranking.")
    for c in categorical_cols:
        if X[c].nunique() < 2:
            print(f"NOTE: '{c}' has a single level; the model cannot learn its effect.")

    models = build_models(quick)
    baseline = DummyRegressor(strategy="mean")

    # ---------------- 1. Nested CV (the ONLY ranking) --------------------- #
    print("\n=== NESTED CROSS-VALIDATION (model selection) ===")
    print(f"Outer: {min(OUTER_SPLITS, len(df))}-fold x {OUTER_REPEATS} repeats | "
          f"Inner: {INNER_SPLITS}-fold x {INNER_REPEATS} grid search | metric: RMSE")

    summary_rows, oof_preds, oof_repeats, fold_tables = [], {}, {}, []

    m, oofs, _ = nested_cv("Baseline (mean)",
                           make_pipeline(baseline, numeric_cols, categorical_cols), None, X, y)
    oof_repeats["Baseline (mean)"] = oofs
    oof_preds["Baseline (mean)"] = np.mean(oofs, axis=0)
    summary_rows.append({"Model": "Baseline (mean)",
                         **{f"{c} mean": m[c].mean() for c in m},
                         **{f"{c} std": m[c].std() for c in m}})

    for name, (reg, grid) in models.items():
        t0 = time.time()
        pipe = make_pipeline(reg, numeric_cols, categorical_cols)
        n_comb = int(np.prod([len(v) for v in grid.values()]))
        print(f"\n[{name}] {n_comb} grid combinations ...")
        m, oofs, folds = nested_cv(name, pipe, grid, X, y)
        oof_repeats[name] = oofs
        oof_preds[name] = np.mean(oofs, axis=0)
        fold_tables.append(folds)
        row = {"Model": name}
        for c in m:
            row[f"{c} mean"] = m[c].mean()
            row[f"{c} std"] = m[c].std()
        summary_rows.append(row)
        print(f"  RMSE {row['RMSE mean']:.1f} +/- {row['RMSE std']:.1f} Pa | "
              f"MAE {row['MAE mean']:.1f} | R2 {row['R2 mean']:.3f} +/- {row['R2 std']:.3f} "
              f"({time.time() - t0:.0f}s)")

    summary = pd.DataFrame(summary_rows)
    ranked = summary[summary["Model"].isin(models.keys())].sort_values("RMSE mean").reset_index(drop=True)
    best_name = ranked.loc[0, "Model"]

    # ---- paired Wilcoxon test: is the winner really better? -------------- #
    best_err = np.abs(y.values - oof_preds[best_name])
    pvals = {}
    for nm in models:
        if nm == best_name:
            pvals[nm] = np.nan
            continue
        diff = np.abs(y.values - oof_preds[nm]) - best_err
        try:
            pvals[nm] = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        except ValueError:
            pvals[nm] = 1.0
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    ranked["p-value vs best"] = ranked["Model"].map(pvals)
    ranked["Distinguishable from best?"] = ranked.apply(
        lambda r: "-" if r["Model"] == best_name else ("yes" if r["p-value vs best"] < ALPHA else "no (tie)"),
        axis=1)

    print("\n--- NESTED CV RANKING (this is the ONLY ranking used) ---")
    show = ranked[["Rank", "Model", "RMSE mean", "RMSE std", "MAE mean", "R2 mean",
                   "p-value vs best", "Distinguishable from best?"]]
    print(show.round(3).to_string(index=False))
    ties = ranked[ranked["Distinguishable from best?"] == "no (tie)"]["Model"].tolist()
    if ties:
        print(f"\nNOTE: statistically tied with {best_name}: {', '.join(ties)}. "
              f"{best_name} is chosen because it has the lowest nested-CV RMSE.")

    summary.to_excel(f"{results_dir}/nested_cv_summary.xlsx", index=False)
    fold_all = pd.concat(fold_tables) if fold_tables else pd.DataFrame()
    if not fold_all.empty:
        fold_all.to_excel(f"{results_dir}/nested_cv_fold_details.xlsx", index=False)
    stability = param_stability(fold_all) if not fold_all.empty else pd.DataFrame()

    # ---------------- 2. Final tuning on ALL data (same scheme/metric) ----- #
    print("\n=== FINAL HYPER-PARAMETER TUNING ON FULL DATA ===")
    print("(used only to choose hyper-parameters, NOT to rank models)")
    final_cv = RepeatedKFold(n_splits=min(FINAL_SPLITS, len(df)),
                             n_repeats=FINAL_REPEATS, random_state=SEED)
    finals, final_rows = {}, []
    for name, (reg, grid) in models.items():
        pipe = make_pipeline(reg, numeric_cols, categorical_cols)
        gs = GridSearchCV(pipe, grid, cv=final_cv, scoring=GRID_SCORING,
                          n_jobs=N_JOBS, refit=True)
        gs.fit(X, y)
        finals[name] = gs.best_estimator_
        res = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
        res.to_excel(f"{results_dir}/gridsearch_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.xlsx",
                     index=False)
        bp = clean_params(gs.best_params_)
        r = ranked.loc[ranked.Model == name].iloc[0]
        final_rows.append({
            "Model": name,
            "Nested-CV rank": int(r["Rank"]),
            "Nested-CV RMSE (expected error, Pa)": r["RMSE mean"],
            "Nested-CV R2": r["R2 mean"],
            "Tuning RMSE on all data (optimistic, info only)": float(np.sqrt(-gs.best_score_)),
            "Best params": str(bp),
        })
        print(f"[{name}] tuned params: {bp}")
    final_df = pd.DataFrame(final_rows).sort_values("Nested-CV rank")
    final_df.to_excel(f"{results_dir}/final_model_summary.xlsx", index=False)

    with pd.ExcelWriter(f"{results_dir}/model_selection_report.xlsx") as xw:
        ranked.to_excel(xw, sheet_name="Ranking (nested CV)", index=False)
        final_df.to_excel(xw, sheet_name="Tuned params", index=False)
        if not stability.empty:
            stability.to_excel(xw, sheet_name="Param stability", index=False)

    # ---------------- 3. Best model = nested-CV winner --------------------- #
    best = ranked.iloc[0]
    print(f"\n>>> BEST MODEL: {best_name} | expected error (nested CV): "
          f"RMSE {best['RMSE mean']:.0f} +/- {best['RMSE std']:.0f} Pa, "
          f"MAE {best['MAE mean']:.0f} Pa, R2 {best['R2 mean']:.3f}")
    info = {
        "best_model": best_name,
        "best_params": final_df.loc[final_df.Model == best_name, "Best params"].iloc[0],
        "nested_cv_RMSE_mean": float(best["RMSE mean"]),
        "nested_cv_RMSE_std": float(best["RMSE std"]),
        "nested_cv_MAE_mean": float(best["MAE mean"]),
        "nested_cv_R2_mean": float(best["R2 mean"]),
        "statistically_tied_with": ties,
        "n_samples": int(len(df)),
        "features": numeric_cols + categorical_cols,
    }
    with open(f"{results_dir}/best_model_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ---------------- 4. Plots -------------------------------------------- #
    names = list(models.keys())
    palette = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
    colors = {nm: palette[i % len(palette)] for i, nm in enumerate(names)}
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))

    # Material column: each material gets its own marker shape AND colour, the same
    # in every panel, with a legend. (Falls back to one style if there is no
    # categorical column.)
    mat_col = "Material" if "Material" in X.columns else (categorical_cols[0] if categorical_cols else None)
    mat_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    mat_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if mat_col is not None:
        materials = sorted(X[mat_col].astype(str).unique())
        mat_style = {m_: (mat_markers[i % len(mat_markers)], mat_colors[i % len(mat_colors)])
                     for i, m_ in enumerate(materials)}
        mat_values = X[mat_col].astype(str).values

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
    lo, hi = y.min() * 0.95, y.max() * 1.05
    for ax in axes.ravel():
        ax.axis("off")
    for ax, nm in zip(axes.ravel(), names):
        ax.axis("on")
        r = summary.loc[summary.Model == nm].iloc[0]
        if mat_col is not None:
            for m_ in materials:
                idx = mat_values == m_
                marker, col = mat_style[m_]
                ax.scatter(y[idx], oof_preds[nm][idx], marker=marker, color=[col], s=70,
                           alpha=0.85, edgecolor="k", label=f"{m_}")
        else:
            ax.scatter(y, oof_preds[nm], alpha=0.75, color=colors[nm], edgecolor="k")
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
        ax.set_xlabel("Actual Delta P (Pa)"); ax.set_ylabel("Predicted Delta P (Pa)")
        star = "  [BEST]" if nm == best_name else ""
        ax.set_title(f"{nm}{star}\nR2={r['R2 mean']:.3f}  RMSE={r['RMSE mean']:.0f} Pa")
        ax.grid(True)
        ax.legend(title=mat_col if mat_col is not None else None, loc="upper left", fontsize=9)
    fig.suptitle("Nested-CV out-of-fold predictions (mean over repeats)", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{results_dir}/prediction_plot.jpg", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        ax.bar(summary["Model"], summary[f"{metric} mean"], yerr=summary[f"{metric} std"],
               capsize=5, color="steelblue", edgecolor="k")
        ax.set_title(metric + (" (Pa)" if metric != "R2" else ""))
        ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.4)
    fig.suptitle("Model comparison (nested CV, mean +/- std over repeats)")
    fig.tight_layout(); fig.savefig(f"{results_dir}/model_comparison.jpg", dpi=200); plt.close(fig)

    # Permutation importance works for EVERY model type (incl. Ridge / SVR) and
    # gives one bar per original feature (the one-hot columns are not split).
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    imp_rows = []
    for ax, nm in zip(axes.ravel(), names):
        ax.axis("on")
        pi = permutation_importance(finals[nm], X, y, scoring="neg_root_mean_squared_error",
                                    n_repeats=30, random_state=SEED)
        order = np.argsort(pi.importances_mean)
        ax.barh(np.array(X.columns)[order], pi.importances_mean[order],
                xerr=pi.importances_std[order], color=colors[nm], edgecolor="k")
        ax.set_title(f"{nm} - permutation importance (RMSE increase, Pa)")
        ax.grid(axis="x", alpha=0.4)
        for c, mu, sd in zip(X.columns, pi.importances_mean, pi.importances_std):
            imp_rows.append({"Model": nm, "Feature": c, "Importance (Pa)": mu, "Std": sd})
    fig.tight_layout(); fig.savefig(f"{results_dir}/feature_importance.jpg", dpi=200); plt.close(fig)
    pd.DataFrame(imp_rows).to_excel(f"{results_dir}/feature_importance_values.xlsx", index=False)
    print(f"Plots and tables saved to '{results_dir}/'")

    # ---------------- 5. Save models -------------------------------------- #
    try:
        for nm, est in finals.items():
            safe = nm.replace(" ", "_").replace("(", "").replace(")", "").lower()
            joblib.dump(est, os.path.join(os.path.dirname(model_file) or ".", f"pressure_model_{safe}.pkl"))
        joblib.dump(finals[best_name], model_file)
        joblib.dump(finals[best_name], "ml_model/pressure_model.pkl")
        print(f"Best model ({best_name}) saved to {model_file} and ml_model/pressure_model.pkl")
    except Exception as e:
        print(f"Error saving model: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/ml_data/cleaned_data.xlsx")
    ap.add_argument("--model", default="models/ml/pressure_model.pkl")
    ap.add_argument("--results", default="test_results")
    ap.add_argument("--quick", action="store_true", help="fast smoke test with tiny grids")
    a = ap.parse_args()
    train_pressure_model(a.input, a.model, a.results, a.quick)