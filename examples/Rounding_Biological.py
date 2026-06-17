import os
import glob
import time
import numpy as np
import pandas as pd

from PolyRound.api import PolyRoundApi
from PolyRound.mutable_classes.polytope import Polytope
from PolyRound.settings import PolyRoundSettings
from dingo import PolytopeSampler

from evaluate_rounding import evaluate_rounding_quality

# This script takes the SIMPLIFIED biological polytopes from BIGG and transforms 
# them to null space and rounds them. In the end, we sample and print some metrics.


# CONFIGURATION
INPUT_DIR = "models_simplified"
OUTPUT_DIR = "models_rounded"
PRINT_CONSOLE_RESULTS = False

def transform_and_round():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize benchmark file taht will save the results
    bench_file = "biology_benchmarks.csv"
    if not os.path.exists(bench_file):
        with open(bench_file, "w") as f:
            f.write(
                "Model,Method,Dimension,Time_sec,T_cond,Cov_ratio,ESS_min,"
                "Cov_trace,Diag_mean,Diag_min,Diag_max,Diag_std,"
                "Dist_I,NonDiag_std\n"
            )

    # Find all simplified A matrices
    a_files = glob.glob(os.path.join(INPUT_DIR, "*_A.csv"))
    
    if not a_files:
        print(f"No *_A.csv files found in {INPUT_DIR}.")
        return

    print(f"Found {len(a_files)} models to process.\n")

    for file_path in a_files:
        base_name = os.path.basename(file_path).replace("_A.csv", "")

        # Only process selected models. Choose whatever you like
        allowed_models = ["iJN746","iAF1260","iAT_PLT_636","iSDY_1059"]
        #allowed_models = ["iAF1260"]
        #allowed_models = ["iJN746"]

        if base_name not in allowed_models:
            print(f"Skipping: {base_name}")
            continue
        
        # Paths for the rest of the matrices
        b_file = os.path.join(INPUT_DIR, f"{base_name}_b.csv")
        s_file = os.path.join(INPUT_DIR, f"{base_name}_S.csv")
        h_file = os.path.join(INPUT_DIR, f"{base_name}_h.csv")

        print(f"\n{'='*60}")
        print(f"Processing: {base_name}")

        try:

            # 1) Load matrices
            print("  1. Loading simplified matrices...")
            A_mat = pd.read_csv(file_path, header=None).values
            b_vec = pd.read_csv(b_file, header=None).values.flatten()
            
            S_mat = None
            h_vec = None
            if os.path.exists(s_file) and os.path.exists(h_file):
                S_mat = pd.read_csv(s_file, header=None).values
                h_vec = pd.read_csv(h_file, header=None).values.flatten()

            # Create PolyRound Polytope object
            poly = Polytope(A_mat, b_vec, S=S_mat, h=h_vec)

            # 2) Transform (project out equalities)
            print("  2. Transforming polytope (removing equalities)...")
            
            settings = PolyRoundSettings()
            settings.hp_flags['FeasibilityTol'] = 1e-6
            settings.hp_flags['OptimalityTol']  = 1e-6
            settings.numerics_threshold = 1e-9
            
            start_t = time.time()
            trans_poly = PolyRoundApi.transform_polytope(poly, settings)
            print(f"     Transformation done in {time.time() - start_t:.2f}s")
            
            A_trans = trans_poly.A.values if hasattr(trans_poly.A, 'values') else trans_poly.A
            b_trans = trans_poly.b.values.flatten() if hasattr(trans_poly.b, 'values') else trans_poly.b.flatten()

            A_trans = np.ascontiguousarray(A_trans, dtype=np.float64)
            b_trans = np.ascontiguousarray(b_trans, dtype=np.float64)

            print(f"     New Dimension (Variables): {A_trans.shape[1]}")
            print(f"     New Dimension (Constraints): {A_trans.shape[0]}")

            # ************ 3) ROUNDING LOOP ***************
            methods = [
                "log_barrier", 
                "vaidya_barrier", 
                "volumetric_barrier" ,
                "john_position" ,
                "min_ellipsoid",
                "isotropic_position"
            ]

            for method_name in methods:
                print(f"  3. Rounding using Dingo ({method_name})...")
                
                start_r = time.time()
                try:
                    A_r, b_r, T_dingo, T_shift_dingo = PolytopeSampler.round_polytope(
                        A_trans, b_trans, method=method_name
                    )
                    elapsed_r = time.time() - start_r
                    
                    print(f"     [SUCCESS] Rounded with {method_name} in {elapsed_r:.2f}s. Evaluating quality...")

                    metrics = evaluate_rounding_quality(A_r, b_r, T_matrix=T_dingo,verbose=PRINT_CONSOLE_RESULTS)
                    
                    # Print metrics to console if you want
                    if PRINT_CONSOLE_RESULTS:
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
                        print(f"       Non-Diag Std Dev     : {metrics['non_diag_std']:.4e} (Min: {metrics['non_diag_min']:.4g}, Max: {metrics['non_diag_max']:.4g})")

                    # Export results to file 
                    out_base = os.path.join(OUTPUT_DIR, f"{base_name}_{method_name}")
                    
                    np.savetxt(f"{out_base}_A.csv", A_r, delimiter=",")
                    np.savetxt(f"{out_base}_b.csv", b_r, delimiter=",")
                    np.savetxt(f"{out_base}_T.csv", T_dingo, delimiter=",")
                    np.savetxt(f"{out_base}_shift.csv", T_shift_dingo, delimiter=",")

                    # Append to global CSV 
                    with open(bench_file, "a") as f:
                        f.write(
                            f"{base_name},{method_name},{A_trans.shape[1]},{elapsed_r:.4f},"
                            f"{metrics['T_cond']:.4e},{metrics['cov_ratio']:.4e},{metrics['ess_min']:.4f},"
                            f"{metrics['cov_trace']:.4e},"
                            f"{metrics['diag_mean']:.4e},{metrics['diag_min']:.4e},"
                            f"{metrics['diag_max']:.4e},{metrics['diag_std']:.4e},"
                            f"{metrics['dist_to_I_fro']:.4e},{metrics['non_diag_std']:.4e}\n"
                        )

                except Exception as e:
                    print(f"     [FAILED] {method_name}: {e}")
                    continue 

        except Exception as e:
            print(f"  [ERROR] Failed on {base_name}: {e}")

if __name__ == "__main__":
    transform_and_round()
