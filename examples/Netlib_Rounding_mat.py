import os
import glob
import time
import numpy as np
import pandas as pd

from evaluate_rounding import evaluate_rounding_quality
from PolyRound.api import PolyRoundApi
from PolyRound.mutable_classes.polytope import Polytope
from PolyRound.settings import PolyRoundSettings
from dingo import PolytopeSampler

# These mat files were downloaded from https://sparse.tamu.edu/LPnetlib.
# In this script we round the already simpplified and transformed polytope . See Simplify_mat_netlib.py

# --- CONFIGURATION ---
INPUT_DIR = "netlib_mat_output"
OUTPUT_DIR = "netlib_rounded_output"
BENCH_FILE = "Netlib_mat_benchmark.csv"
DIR_ROUNDED_SAMPLES = "netlib_rounded_samples"
DIR_TRANSFORMED_SAMPLES = "netlib_transformed_samples"
SAVE_ROUNDED_SAMPLES = True     
SAVE_TRANSFORMED_SAMPLES = True
VERIFY_SAMPLES = True
PRINT_CONSOLE_RESULTS = False

ALLOWED_MODELS = [
    #"lp_afiro",
    #"lp_adlittle",
    #"lp_etamacro",
    #"lp_scorpion",
    #"lp_sierra",
    #"lp_degen2",
    #"lp_cycle",
    #"lp_degen3",
    #"lp_stocfor2",
    #"lp_beaconfd",
    #"lp_bore3d",
    #"lp_grow22",
    #"lp_blend",
    #"lp_stocfor1",
    #"lp_25fv47",
    "lp_brandy",
    #"lp_agg2",
    #"lp_bnl2",
    #"lp_agg",
    #"lp_bandm"
]

def run_rounding_benchmark():
 
    for d in [OUTPUT_DIR, DIR_ROUNDED_SAMPLES, DIR_TRANSFORMED_SAMPLES]:
        if not os.path.exists(d):
            os.makedirs(d)

    if not os.path.exists(BENCH_FILE):
        with open(BENCH_FILE, "w") as f:
            f.write(
                "Model,Method,Dimension,Time_sec,Sampling_sec,T_cond,Cov_ratio,ESS_min,"
                "Cov_trace,Diag_mean,Diag_min,Diag_max,Diag_std,"
                "Dist_I,"
                "NonDiag_mean,NonDiag_min,NonDiag_max,NonDiag_std\n"
            )
        print(f"--- Created new benchmark file: {BENCH_FILE} ---")

    A_files = glob.glob(os.path.join(INPUT_DIR, "*_A.csv"))
    
    if not A_files:
        print(f"No _A.csv files found in {INPUT_DIR}")
        return

    print(f"Found {len(A_files)} total models in {INPUT_DIR}.\n")

    settings = PolyRoundSettings()
    # settings.hp_flags['FeasibilityTol'] = 1e-6
    # settings.hp_flags['OptimalityTol']  = 1e-6
    # settings.numerics_threshold = 1e-6
    settings.verbose = True

    for file_A in A_files:
        # extract base name (e.g., "netlib_mat_output/lp_degen2_A.csv" -> "lp_degen2")
        base_name = os.path.basename(file_A).replace("_A.csv", "")

        # apply filtering
        if ALLOWED_MODELS is not None and base_name not in ALLOWED_MODELS:
            continue

        print(f"--- Processing Model: {base_name} ---")

        file_b = os.path.join(INPUT_DIR, f"{base_name}_b.csv")
        
        if not os.path.exists(file_b):
            print(f"  [!] Error: Matching _b.csv missing for {base_name}. Skipping.")
            continue

        try:
            A_trans = np.ascontiguousarray(pd.read_csv(file_A, header=None, dtype=np.float64).values)
            b_trans = np.ascontiguousarray(pd.read_csv(file_b, header=None, dtype=np.float64).values.flatten())
            
            # ************ 3) ROUNDING LOOP ***************
            methods = [
                #"log_barrier", 
                #"vaidya_barrier", 
                #"volumetric_barrier",
                #"john_position", 
                #"min_ellipsoid",
                # "isotropic_position",
                "polyround"
            ]

            for method_name in methods:
                print(f"  -> Rounding using Dingo ({method_name})...")
                
                try:
                    if method_name == "polyround":
                        pr_polytope = Polytope(pd.DataFrame(A_trans), pd.Series(b_trans))
                        
                        start_r = time.time()
                        rounded_pr_poly = PolyRoundApi.round_polytope(pr_polytope, settings=settings)
                        elapsed_r = time.time() - start_r
                        
                        A_r = rounded_pr_poly.A.values
                        b_r = rounded_pr_poly.b.values.flatten()
                        T_matrix = rounded_pr_poly.transformation.values
                        T_shift = rounded_pr_poly.shift.values
                    else:
                        # Dingo rounding
                        start_r = time.time()
                        A_r, b_r, T_matrix, T_shift = PolytopeSampler.round_polytope(
                            A_trans, b_trans, method=method_name
                        )
                        elapsed_r = time.time() - start_r
                    
                    print(f"     [SUCCESS] Rounded with {method_name} in {elapsed_r:.2f}s. Evaluating quality...")

                    metrics, samples_array = evaluate_rounding_quality(A_r, b_r, T_matrix=T_matrix, verbose=PRINT_CONSOLE_RESULTS)
                    
                    # Print metrics to console if you want
                    if PRINT_CONSOLE_RESULTS:
                        print(f"       Sampling Time        : {metrics['sampling_time']:.2f}s")
                        print(f"       T condition number   : {metrics['T_cond']:.4g}")
                        print(f"       Cov eigenvalue ratio : {metrics['cov_ratio']:.4g}")
                        print(f"       Min ESS              : {metrics['ess_min']:.1f}")
                        print(f"       Covariance Trace     : {metrics['cov_trace']:.4g} (Expected: ~{A_trans.shape[1]})")
                        print(
                            f"       Diag elements : Mean={metrics['diag_mean']:.4g}, "
                            f"Min={metrics['diag_min']:.4g}, "
                            f"Max={metrics['diag_max']:.4g}, "
                            f"Std={metrics['diag_std']:.4g}"
                        )
                        print(f"       Dist to Identity (Fro): {metrics['dist_to_I_fro']:.4g} (Avg Error: {metrics['dist_to_I_mean']:.4g})")
                        print(
                            f"       Non-Diag elements : Mean={metrics['non_diag_mean']:.4e}, "
                            f"Std={metrics['non_diag_std']:.4e}, "
                            f"Min={metrics['non_diag_min']:.4g}, "
                            f"Max={metrics['non_diag_max']:.4g}"
                        )
                    
                    # Export rounded results to output folder 
                    out_base = os.path.join(OUTPUT_DIR, f"{base_name}_{method_name}")
                    
                    np.savetxt(f"{out_base}_A.csv", A_r, delimiter=",")
                    np.savetxt(f"{out_base}_b.csv", b_r, delimiter=",")
                    np.savetxt(f"{out_base}_T.csv", T_matrix, delimiter=",")
                    np.savetxt(f"{out_base}_shift.csv", T_shift, delimiter=",")

                    # Append to global CSV 
                    with open(BENCH_FILE, "a") as f:
                        f.write(
                            f"{base_name},{method_name},{A_trans.shape[1]},{elapsed_r:.4f},{metrics['sampling_time']:.4f},"
                            f"{metrics['T_cond']:.4e},{metrics['cov_ratio']:.4e},{metrics.get('min_eigenvalue', 0):.4e},{metrics['ess_min']:.4f},"
                            f"{metrics['cov_trace']:.4e},"
                            f"{metrics['diag_mean']:.4e},{metrics['diag_min']:.4e},"
                            f"{metrics['diag_max']:.4e},{metrics['diag_std']:.4e},"
                            f"{metrics['dist_to_I_fro']:.4e},"
                            f"{metrics['non_diag_mean']:.4e},{metrics['non_diag_min']:.4e},"
                            f"{metrics['non_diag_max']:.4e},{metrics['non_diag_std']:.4e}\n"
                        )
                    
                    # SAVING ROUNDED SAMPLES 
                    if SAVE_ROUNDED_SAMPLES:
                        out_rounded_file = os.path.join(DIR_ROUNDED_SAMPLES, f"{base_name}_{method_name}.csv")
                        np.savetxt(out_rounded_file, samples_array, delimiter=",", fmt="%.5f")
                        print(f"     [INFO] Saved {samples_array.shape[0]} rounded samples to {DIR_ROUNDED_SAMPLES}")

                    # SAVING / VERIFYING TRANSFORMED SAMPLES
                    if SAVE_TRANSFORMED_SAMPLES or VERIFY_SAMPLES:
                        try:
                            # Reverse Transform: x_trans = T_dingo * x_rounded + shift
                            X_trans = samples_array @ T_matrix.T + T_shift.flatten()
                            
                            if VERIFY_SAMPLES:
                                # Check if samples satisfy the intermediate polytope: A_trans * x <= b_trans
                                violations = (X_trans @ A_trans.T) - b_trans
                                tol = 1e-6
                                valid_samples_mask = np.all(violations <= tol, axis=1)
                                num_valid = np.sum(valid_samples_mask)
                                num_total = X_trans.shape[0]
                                
                                if num_valid == num_total:
                                    print(f"     [VERIFY] SUCCESS: All {num_total} mapped samples satisfy A_trans constraints (tol={tol}).")
                                else:
                                    max_violation = np.max(violations)
                                    print(f"     [VERIFY] WARNING: {num_total - num_valid} out of {num_total} samples violate A_trans constraints!")
                                    print(f"              Max constraint violation: {max_violation:.4e}")
                                    
                            if SAVE_TRANSFORMED_SAMPLES:
                                out_transformed_file = os.path.join(DIR_TRANSFORMED_SAMPLES, f"{base_name}_{method_name}.csv")
                                np.savetxt(out_transformed_file, X_trans, delimiter=",", fmt="%.5f")
                                print(f"     [INFO] Saved {X_trans.shape[0]} mapped-back original samples to {DIR_TRANSFORMED_SAMPLES}")
                        except Exception as e:
                            print(f"     [ERROR] Failed to map back or verify samples: {e}")
                except Exception as e:
                    print(f"     [FAILED] {method_name}: {e}")
                    continue 

        except Exception as e:
            print(f"  [ERROR] Failed processing matrices for {base_name}: {e}")
        
        print("")
        
    print("--- Rounding pipeline complete! ---")

if __name__ == "__main__":
    run_rounding_benchmark()