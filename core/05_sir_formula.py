import math
import numpy as np
from scipy import integrate

# --- Material Profiles ---
MATERIALS = {
    "plastic bead": {
        "rho_p": 935.0,
        "rho_b": 528.0,
        "mu_w": 0.56,
        "phi_e_rad": 0.86,
        "c_w": 0.0,
        "d_m": 4.75e-3
    },
    "potash": {
        "rho_p": 2046.0,
        "rho_b": 1196.0,
        "mu_w": 0.40,
        "phi_e_rad": 0.76,
        "c_w": 0.0,
        "d_m": 1.0e-3
    },
    "zeolite": {
        "rho_p": 2210.0,
        "d_m": 2.20e-3,
        "rho_b": 2000.0,
        "mu_w": 0.50,
        "phi_e_rad": 0.80,
        "c_w": 0.0
    }
}

G = 9.81
RHO_AIR = 1.204
MU_AIR = 1.825e-5

def compute_pressure_drop():
    print("=== Plug-2 Pressure Drop Calculator ===\n")
    
    mat_choice = input("Enter material (plastic bead, potash, zeolite): ").strip().lower()
    if mat_choice not in MATERIALS:
        print("Invalid material. Exiting.")
        return
    mat = MATERIALS[mat_choice]
    
    if mat["rho_b"] is None:
        try:
            mat["rho_b"] = float(input("Enter Zeolite Bulk Density (kg/m^3): "))
            mat["mu_w"] = float(input("Enter Zeolite Wall Friction Coefficient (mu_w): "))
            mat["phi_e_rad"] = float(input("Enter Zeolite Effective Internal Friction Angle (radians): "))
        except ValueError:
            print("Invalid input.")
            return

    try:
        pipe_d_mm = float(input("Enter pipe diameter (mm): "))
        discharge_lpm = float(input("Enter discharge rate (L/min): "))
        plug_length = float(input("Enter plug length (meters): "))
        theta_deg = float(input("Enter plug front angle (degrees): "))
        h_mm = float(input("Enter stationary layer height at end (mm): "))
    except ValueError:
        print("Invalid input.")
        return
        
    D = pipe_d_mm / 1000.0
    h = h_mm / 1000.0
    L = plug_length
    theta = math.radians(theta_deg)
    
    print("\n--- Calculating Intermediate Variables ---")
    
    discharge_m3_s = discharge_lpm / 60000.0
    pipe_area = math.pi * (D / 2)**2
    u_sf = discharge_m3_s / pipe_area
    print(f"Superficial Air Velocity (U_sf): {u_sf:.4f} m/s")
    
    ar = (G * RHO_AIR * (mat["rho_p"] - RHO_AIR) * mat["d_m"]**3) / (MU_AIR**2)
    print(f"Archimedes Number (Ar): {ar:.2f}")
    
    u_par = u_sf - 0.042 * (ar ** 0.213)
    print(f"Particle Velocity (U_par): {u_par:.4f} m/s")
    
    u_plu = 0.579 + 0.774 * u_par
    print(f"Plug Velocity (U_plu): {u_plu:.4f} m/s")
    
    mu_e = math.tan(mat["phi_e_rad"])
    k_stress = (1.0 - math.sin(mat["phi_e_rad"])) / (1.0 + math.sin(mat["phi_e_rad"]))
    epsilon = 1.0 - (mat["rho_b"] / mat["rho_p"])
    rho_bst = mat["rho_b"] 
    
    print(f"Effective Friction Tangent (mu_e): {mu_e:.4f}")
    print(f"Stress Transmission Ratio (K): {k_stress:.4f}")
    print(f"Plug Void Fraction (epsilon): {epsilon:.4f}")
    
    acos_inner = max(-1.0, min(1.0, 1.0 - (2.0 * h / D)))
    phi = 2.0 * math.acos(acos_inner)
    alpha = (phi - math.sin(phi)) / (2.0 * math.pi)
    print(f"Stationary Layer Area Fraction (alpha): {alpha:.4f}")
    
    y_cm = h
    def integrand(y):
        val = y * (D - y)
        return math.sqrt(max(val, 0.0)) * (y - y_cm)
    
    integral_a, _ = integrate.quad(integrand, y_cm, D - y_cm)
    part_a = ((8.0 * mu_e * G * rho_bst * h) / (math.pi * (D ** 2) * (D - h))) * integral_a
    print(f"Integral Parameter A component (part_a): {part_a:.2f}")
    
    term1_b = (mat["mu_w"] * mat["rho_b"] * G) / (2.0 * math.tan(theta)) if math.tan(theta) != 0 else 0.0
    term2_b = alpha * rho_bst * (1.0 - alpha * (rho_bst / mat["rho_b"])) * (u_par ** 2)
    term3_inner = (D / 2.0) * ((phi / 2.0) + (math.sin(2.0 * phi) / 4.0) - math.sin(phi)) + h * math.sin(phi)
    term3_b = 2.0 * rho_bst * G * term3_inner
    b = term1_b + term2_b + term3_b
    print(f"Momentum & Resistance Parameter B: {b:.2f}")
    
    exp_term = math.exp((4.0 * mat["mu_w"] * k_stress * L) / D)
    num1 = ((4.0 * mat["mu_w"] * k_stress) / D) * b
    num2 = mat["mu_w"] * mat["rho_b"] * G
    num3 = (4.0 * mat["c_w"]) / D
    numerator = (num1 + num2 + num3) * exp_term - num3 - num2
    
    den1 = ((1.0 - epsilon) * 4.0 * mat["mu_w"] * k_stress) / D
    den2 = (epsilon / L) * (exp_term - 1.0)
    denominator = den1 + den2
    
    part_c = numerator / denominator
    delta_p = part_a + part_c
    
    print("\n--- Final Output ---")
    print(f"Total Pressure Drop (Delta P): {delta_p:.2f} Pa")

if __name__ == "__main__":
    compute_pressure_drop()