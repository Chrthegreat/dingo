import os
import time
import numpy as np
import pandas as pd

from dingo import PolytopeSampler
from evaluate_rounding import evaluate_rounding_quality

from PolyRound.api import PolyRoundApi
from PolyRound.mutable_classes.polytope import Polytope as PRPolytope
from PolyRound.settings import PolyRoundSettings

# CONFIGURATION
OUTPUT_DIR = "skinny_rounded"
PRINT_CONSOLE_RESULTS = False
SAVE_ROUNDED_SAMPLES = True      
SAVE_TRANSFORMED_SAMPLES = False  
VERIFY_SAMPLES = True           
DIR_ROUNDED_SAMPLES = "skinny_rounded_samples"
DIR_TRANSFORMED_SAMPLES = "skinny_transformed_samples"


def generate_skinny_cube(dim: int, skinny_percentage: float):
    """
    Python translation of generate_skinny_cube.
    Creates a skinny cube of given dimension in H-representation (Ax <= b).
    
    :param dim: Dimension of the polytope
    :param skinny_percentage: Fraction of dimensions to be 'stretched' (e.g., 0.10 for 10%)
    :return: A, b (numpy arrays)
    """
    skinny_count = int(dim * skinny_percentage)

    if skinny_percentage > 0 and skinny_count == 0:
        skinny_count = 1
        
    bounds = np.ones(dim)
    bounds[:skinny_count] = 100.0

    # Top half: x_i <= bounds[i]  -> A_top is Identity
    # Bottom half: -x_i <= bounds[i] -> A_bottom is -Identity
    A = np.vstack([np.eye(dim), -np.eye(dim)])
    b = np.concatenate([bounds, bounds])
    
    return A, b


def run_benchmark():
    for d in [OUTPUT_DIR, DIR_ROUNDED_SAMPLES, DIR_TRANSFORMED_SAMPLES]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Benchmark file
    bench_file = "skinny_benchmark.csv"
    if not os.path.exists(bench_file):
        with open(bench_file, "w") as f:
            f.write(
                "Model,Method,Dimension,Time_sec,Sampling_sec,T_cond,Cov_ratio,Min_eig,ESS_min,"
                "Cov_trace,Diag_mean,Diag_min,Diag_max,Diag_std,"
                "Dist_I,"
                "NonDiag_mean,NonDiag_min,NonDiag_max,NonDiag_std\n"
            )

    dims_to_test = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    skinny_percentages = [0.10, 0.50]
    
    methods = [
        "log_barrier", 
        "vaidya_barrier", 
        "volumetric_barrier",
        "john_position", 
        "min_ellipsoid",
        "polyround" 
    ]

    for dim in dims_to_test:
        for pct in skinny_percentages:
            model_name = f"SkinnyCube{int(pct * 100)}"
            base_name = f"{model_name}_dim{dim}"

            print(f"\n{'='*60}")
            print(f"Processing: {base_name} ({pct*100}% skinny)")

            try:
                A, b = generate_skinny_cube(dim, pct)
                A = np.ascontiguousarray(A, dtype=np.float64)
                b = np.ascontiguousarray(b, dtype=np.float64)

                print(f"     Dimension (Variables): {A.shape[1]}")
                print(f"     Dimension (Constraints): {A.shape[0]}")

                # ************ ROUNDING LOOP ***************
                for method_name in methods:
                    print(f"  Rounding using {method_name}...")
                    
                    try:
                        if method_name == "polyround":
                          
                            pr_polytope = PRPolytope(pd.DataFrame(A), pd.Series(b))
                            settings = PolyRoundSettings()

                            start_r = time.time()
                            rounded_pr_poly = PolyRoundApi.round_polytope(pr_polytope, settings=settings)
                            elapsed_r = time.time() - start_r
                            
                            # Extract attributes back to numpy arrays
                            A_r = rounded_pr_poly.A.values
                            b_r = rounded_pr_poly.b.values.flatten()
                            T_matrix = rounded_pr_poly.transformation.values
                            T_shift = rounded_pr_poly.shift.values
                        else:
                            # Dingo rounding
                            start_r = time.time()
                            A_r, b_r, T_matrix, T_shift = PolytopeSampler.round_polytope(
                                A, b, method=method_name
                            )
                            elapsed_r = time.time() - start_r
                        
                        print(f"     [SUCCESS] Rounded with {method_name} in {elapsed_r:.2f}s. Evaluating quality...")

                        metrics, samples_array = evaluate_rounding_quality(
                            A_r, b_r, 
                            T_matrix=T_matrix, 
                            verbose=PRINT_CONSOLE_RESULTS
                        )
                        
                        if PRINT_CONSOLE_RESULTS:
                            print(f"       Sampling Time        : {metrics['sampling_time']:.2f}s")
                            print(f"       T condition number   : {metrics['T_cond']:.4g}")
                            print(f"       Cov eigenvalue ratio : {metrics['cov_ratio']:.4g}")
                            print(f"       Min ESS              : {metrics['ess_min']:.1f}")
                            
                        # Export results to file 
                        out_base = os.path.join(OUTPUT_DIR, f"{base_name}_{method_name}")
                        
                        np.savetxt(f"{out_base}_A.csv", A_r, delimiter=",")
                        np.savetxt(f"{out_base}_b.csv", b_r, delimiter=",")
                        np.savetxt(f"{out_base}_T.csv", T_matrix, delimiter=",")
                        np.savetxt(f"{out_base}_shift.csv", T_shift, delimiter=",")

                        # SAVING ROUNDED SAMPLES
                        if SAVE_ROUNDED_SAMPLES:
                            out_rounded_file = os.path.join(DIR_ROUNDED_SAMPLES, f"{base_name}_{method_name}.csv")
                            np.savetxt(out_rounded_file, samples_array, delimiter=",", fmt="%.5f")
                            print(f"     [INFO] Saved {samples_array.shape[0]} rounded samples.")

                        # SAVING / VERIFYING TRANSFORMED SAMPLES
                        if SAVE_TRANSFORMED_SAMPLES or VERIFY_SAMPLES:
                            try:
                                # Reverse transform: x_trans = T_matrix * x_rounded + shift
                                X_trans = samples_array @ T_matrix.T + T_shift.flatten()
                                
                                if SAVE_TRANSFORMED_SAMPLES:
                                    out_transformed_file = os.path.join(DIR_TRANSFORMED_SAMPLES, f"{base_name}_{method_name}.csv")
                                    np.savetxt(out_transformed_file, X_trans, delimiter=",", fmt="%.5f")
                                    
                                if VERIFY_SAMPLES:
                                    # Check if samples satisfy original polytope: A * x <= b
                                    violations = (X_trans @ A.T) - b
                                    tol = 1e-6
                                    valid_samples_mask = np.all(violations <= tol, axis=1)
                                    num_valid = np.sum(valid_samples_mask)
                                    num_total = X_trans.shape[0]
                                    
                                    if num_valid == num_total:
                                        print(f"     [VERIFY] SUCCESS: All {num_total} mapped samples satisfy original constraints.")
                                    else:
                                        max_violation = np.max(violations)
                                        print(f"     [VERIFY] WARNING: {num_total - num_valid}/{num_total} samples violate constraints! Max violation: {max_violation:.4e}")
                                        
                            except Exception as e:
                                print(f"     [ERROR] Failed to map back or verify samples: {e}")

                        # Append to global CSV 
                        with open(bench_file, "a") as f:
                            # Append to global CSV 
                            with open(bench_file, "a") as f:
                                f.write(
                                    f"{model_name},{method_name},{A.shape[1]},{elapsed_r:.4f},{metrics['sampling_time']:.4f},"
                                    f"{metrics['T_cond']:.4e},{metrics['cov_ratio']:.4e},{metrics['min_eigenvalue']:.4e},{metrics['ess_min']:.4f},"
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
    run_benchmark()