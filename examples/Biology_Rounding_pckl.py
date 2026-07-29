import os
import glob
import time
import pickle
import numpy as np
import pandas as pd

from dingo import PolytopeSampler
from evaluate_rounding import evaluate_rounding_quality

from PolyRound.api import PolyRoundApi
from PolyRound.settings import PolyRoundSettings
from PolyRound.api import Polytope as PRPolytope

# THESE pckl were downloaded from ZENODO and they are already simplified and transformed
# Thats why we only do the rounding loop.

# CONFIGURATION
INPUT_DIR = "models_pckl" 
OUTPUT_DIR = "models_rounded_pckl"
PRINT_CONSOLE_RESULTS = False
SAVE_ROUNDED_SAMPLES = True       # Save raw samples from the rounded polytope
SAVE_TRANSFORMED_SAMPLES = True   # Save samples mapped back to the original biological space
VERIFY_SAMPLES = True             # Check if mapped-back samples satisfy A_trans * x <= b_trans
DIR_ROUNDED_SAMPLES = "models_rounded_samples"
DIR_TRANSFORMED_SAMPLES = "models_transformed_samples"

def load_and_round():

    for d in [OUTPUT_DIR, DIR_ROUNDED_SAMPLES, DIR_TRANSFORMED_SAMPLES]:
        if not os.path.exists(d):
            os.makedirs(d)

    # benchmark file
    bench_file = "biology_benchmarks_pckl.csv"
    if not os.path.exists(bench_file):
        with open(bench_file, "w") as f:
            f.write(
                "Model,Method,Dimension,Time_sec,Sampling_sec,T_cond,Cov_ratio,Min_eig,ESS_min,"
                "Cov_trace,Diag_mean,Diag_min,Diag_max,Diag_std,"
                "Dist_I,"
                "NonDiag_mean,NonDiag_min,NonDiag_max,NonDiag_std\n"
            )

    # find all .pckl files
    pckl_files = glob.glob(os.path.join(INPUT_DIR, "*.pckl"))
    
    if not pckl_files:
        print(f"No .pckl files found in {INPUT_DIR}.")
        return

    print(f"Found {len(pckl_files)} models to process.\n")

    for file_path in pckl_files:
        base_name = os.path.basename(file_path).replace(".xml.pckl", "").replace("polytope_", "")

        print(f"\n{'='*60}")
        print(f"Processing: {base_name}")

        try:

            print("  Loading Polytope object from tuple...")
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            poly_obj = data[0]

            A_trans = poly_obj.A.values
            b_trans = poly_obj.b.values.flatten()

            A_trans = np.ascontiguousarray(A_trans, dtype=np.float64)
            b_trans = np.ascontiguousarray(b_trans, dtype=np.float64)

            print(f"     Dimension (Variables): {A_trans.shape[1]}")
            print(f"     Dimension (Constraints): {A_trans.shape[0]}")

            # ************ 3) ROUNDING LOOP ***************
            methods = [
                "log_barrier", 
                "vaidya_barrier", 
                "volumetric_barrier",
                "john_position", 
                "min_ellipsoid",
                "polyround"
            ]

            for method_name in methods:
                print(f"  3. Rounding using Dingo ({method_name})...")
                
                try:
                    if method_name == "polyround":
                        pr_polytope = PRPolytope(pd.DataFrame(A_trans), pd.Series(b_trans))
                        settings = PolyRoundSettings()
                        
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
                        A_r, b_r, T_dingo, T_shift_dingo = PolytopeSampler.round_polytope(
                            A_trans, b_trans, method=method_name
                        )
                        elapsed_r = time.time() - start_r

                    print(f"     [SUCCESS] Rounded with {method_name} in {elapsed_r:.2f}s. Evaluating quality...")

                    metrics, samples_array = evaluate_rounding_quality(
                        A_r, b_r, 
                        T_matrix=T_dingo, 
                        verbose=PRINT_CONSOLE_RESULTS
                    )

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
                    
                    # Export results to file 
                    out_base = os.path.join(OUTPUT_DIR, f"{base_name}_{method_name}")
                    
                    np.savetxt(f"{out_base}_A.csv", A_r, delimiter=",")
                    np.savetxt(f"{out_base}_b.csv", b_r, delimiter=",")
                    np.savetxt(f"{out_base}_T.csv", T_dingo, delimiter=",")
                    np.savetxt(f"{out_base}_shift.csv", T_shift_dingo, delimiter=",")

                    # SAVING ROUNDED SAMPLES
                    if SAVE_ROUNDED_SAMPLES:
                        out_rounded_file = os.path.join(DIR_ROUNDED_SAMPLES, f"{base_name}_{method_name}.csv")
                        np.savetxt(out_rounded_file, samples_array, delimiter=",", fmt="%.5f")
                        print(f"     [INFO] Saved {samples_array.shape[0]} rounded samples to {DIR_ROUNDED_SAMPLES}")

                    # SAVING / VERIFYING TRANSFORMED SAMPLES
                    if SAVE_TRANSFORMED_SAMPLES or VERIFY_SAMPLES:
                        try:
                            # Reverse transform: x_trans = T_dingo * x_rounded + shift
                            X_trans = samples_array @ T_dingo.T + T_shift_dingo.flatten()
                            
                            if SAVE_TRANSFORMED_SAMPLES:
                                # Reverse PolyRound's Null-Space transform
                                # X_orig_transposed = poly_obj.back_transform(X_trans.T)
                                
                                # Transpose back to (N_samples, original_dimensions) for saving
                                # X_orig = X_orig_transposed.T 
                                
                                out_transformed_file = os.path.join(DIR_TRANSFORMED_SAMPLES, f"{base_name}_{method_name}.csv")
                                np.savetxt(out_transformed_file, X_trans, delimiter=",", fmt="%.5f")
                                print(f"     [INFO] Saved {X_trans.shape[0]} mapped-back original samples to {DIR_TRANSFORMED_SAMPLES}")
                                
                            if VERIFY_SAMPLES:
                                # Check if samples satisfy original polytope: A_trans * x <= b_trans
                                # We calculate (X_trans @ A_trans.T) - b_trans. Values should be <= 0.
                                violations = (X_trans @ A_trans.T) - b_trans
                                
                                # As usual, use a small tolerance for floating point errors during transformations
                                tol = 1e-6
                                valid_samples_mask = np.all(violations <= tol, axis=1)
                                num_valid = np.sum(valid_samples_mask)
                                num_total = X_trans.shape[0]
                                
                                if num_valid == num_total:
                                    print(f"     [VERIFY] SUCCESS: All {num_total} mapped samples satisfy the original polytope (tol={tol}).")
                                else:
                                    max_violation = np.max(violations)
                                    print(f"     [VERIFY] WARNING: {num_total - num_valid} out of {num_total} samples violate original constraints!")
                                    print(f"              Max constraint violation: {max_violation:.4e}")
                                    
                        except Exception as e:
                            print(f"     [ERROR] Failed to map back or verify samples: {e}")

                    # Append to global CSV 
                    with open(bench_file, "a") as f:
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

                except Exception as e:
                    print(f"     [FAILED] {method_name}: {e}")
                    continue 

        except Exception as e:
            print(f"  [ERROR] Failed on {base_name}: {e}")

if __name__ == "__main__":
    load_and_round()