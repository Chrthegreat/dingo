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

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = "models_biology"                  # Directory with raw .xml files
SIMPLIFIED_DIR = "models_simplified"          # Where to save A, b, S, h matrices
ROUNDED_DIR = "models_rounded"                # Where to save rounded A, b, T, shift matrices

PRINT_CONSOLE_RESULTS = False
SAVE_ROUNDED_SAMPLES = False
SAVE_TRANSFORMED_SAMPLES = False
VERIFY_SAMPLES = True
DIR_ROUNDED_SAMPLES = "models_rounded_samples"
DIR_TRANSFORMED_SAMPLES = "models_transformed_samples"

def full_pipeline():
    # Setup Directories
    for d in [SIMPLIFIED_DIR, ROUNDED_DIR, DIR_ROUNDED_SAMPLES, DIR_TRANSFORMED_SAMPLES]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Setup Benchmark Files
    bench_file_rounding = "biology_benchmarks.csv"
    if not os.path.exists(bench_file_rounding):
        with open(bench_file_rounding, "w") as f:
            f.write(
                "Model,Method,Dimension,Time_sec,Sampling_sec,T_cond,Cov_ratio,Min_eig,ESS_min,"
                "Cov_trace,Diag_mean,Diag_min,Diag_max,Diag_std,"
                "Dist_I,"
                "NonDiag_mean,NonDiag_min,NonDiag_max,NonDiag_std\n"
            )
            
    summary_path_simplify = os.path.join(SIMPLIFIED_DIR, "biology_simplify.csv")
    simplify_results = []

    # Find XML files
    xml_files = glob.glob(os.path.join(INPUT_DIR, "*.xml"))
    if not xml_files:
        print(f"No .xml files found in {INPUT_DIR}")
        return

    print(f"Found {len(xml_files)} biological models to process.\n")

    for file_path in xml_files:
        base_name = os.path.basename(file_path).replace(".xml", "")
        
        # Only process selected models. Choose whatever you like
        allowed_models = [
            "iBWG_1329",
            #"e_coli_core",
            "iML1515",
            "iSB619",
            #"iEC1344_C",
            #"iHN637",
            #"iAT_PLT_636",
            #"iLJ478",
            #"Recon3D",
            #"iYL1228",
            "iJO1366",
            #"RECON1",
            #"iJR904",
            "iJN746",
            #"iAB_RBC_283",
            #"iJN678",
            #"iNF517",
            #"iSDY_1059",
            #"iAF1260"
        ]

        if base_name not in allowed_models:
            print(f"Skipping: {base_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {base_name}")

        try:
            # ***********1: LOAD & SIMPLIFY*****************
            print("  Loading SBML model...")
            poly = PolyRoundApi.sbml_to_polytope(file_path)
            
            settings = PolyRoundSettings()

            # Custom polyround settings
            # settings.hp_flags['FeasibilityTol'] = 1e-6
            # settings.hp_flags['OptimalityTol']  = 1e-6
            # settings.numerics_threshold = 1e-6
            # settings.verbose = False 

            print("  1: Simplifying polytope...")
            start_t = time.time()
            simplified_poly = PolyRoundApi.simplify_polytope(poly, settings)
            elapsed_simp = time.time() - start_t
            
            orig_cons = poly.A.shape[0] if poly.A is not None else 0
            final_cons = simplified_poly.A.shape[0]
            reduction = 100 * (orig_cons - final_cons) / orig_cons if orig_cons > 0 else 0
            print(f"     Simplification done in {elapsed_simp:.2f}s! Constraints: {orig_cons} -> {final_cons} ({reduction:.1f}% reduction)")

            simplify_results.append({
                "Model": base_name,
                "Time (s)": round(elapsed_simp, 2),
                "Initial Constraints": orig_cons,
                "Final Constraints": final_cons,
                "% Reduction": round(reduction, 1),
            })

            # Save Simplified Matrices
            out_A = os.path.join(SIMPLIFIED_DIR, f"{base_name}_A.csv")
            out_b = os.path.join(SIMPLIFIED_DIR, f"{base_name}_b.csv")
            pd.DataFrame(simplified_poly.A).to_csv(out_A, index=False, header=False)
            pd.DataFrame(simplified_poly.b).to_csv(out_b, index=False, header=False)
            
            if simplified_poly.S is not None and simplified_poly.S.size > 0:
                out_S = os.path.join(SIMPLIFIED_DIR, f"{base_name}_S.csv")
                out_h = os.path.join(SIMPLIFIED_DIR, f"{base_name}_h.csv")
                pd.DataFrame(simplified_poly.S).to_csv(out_S, index=False, header=False)
                pd.DataFrame(simplified_poly.h).to_csv(out_h, index=False, header=False)

            # Update simplification summary CSV iteratively
            pd.DataFrame(simplify_results).to_csv(summary_path_simplify, index=False)

            # **************2: TRANSFORM (REMOVE EQUALITIES)*******************
            print("  2: Transforming polytope (removing equalities)...")

            start_t = time.time()
            
            # PolyRound only transforms if there are equality constraints
            if not simplified_poly.inequality_only:
                trans_poly = PolyRoundApi.transform_polytope(simplified_poly, settings)
            else:
                trans_poly = simplified_poly
                
            print(f"     Transformation done in {time.time() - start_t:.2f}s")
            
            A_trans = trans_poly.A.values if hasattr(trans_poly.A, 'values') else trans_poly.A
            b_trans = trans_poly.b.values.flatten() if hasattr(trans_poly.b, 'values') else trans_poly.b.flatten()

            A_trans = np.ascontiguousarray(A_trans, dtype=np.float64)
            b_trans = np.ascontiguousarray(b_trans, dtype=np.float64)

            print(f"     New Dimension (Variables): {A_trans.shape[1]}")
            print(f"     New Dimension (Constraints): {A_trans.shape[0]}")

            # ****************3: ROUNDING LOOP***********************
            methods = [
                #"log_barrier", 
                #"vaidya_barrier", 
                #"volumetric_barrier",
                #"john_position", 
                #"min_ellipsoid",
                #"isotropic_position",
                "polyround"
            ]

            for method_name in methods:
                print(f"  3. Rounding using ({method_name})...")
                
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
                        start_r = time.time()
                        A_r, b_r, T_matrix, T_shift = PolytopeSampler.round_polytope(
                            A_trans, b_trans, method=method_name
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
                    
                    # Export rounded matrices 
                    out_base = os.path.join(ROUNDED_DIR, f"{base_name}_{method_name}")
                    np.savetxt(f"{out_base}_A.csv", A_r, delimiter=",")
                    np.savetxt(f"{out_base}_b.csv", b_r, delimiter=",")
                    np.savetxt(f"{out_base}_T.csv", T_matrix, delimiter=",")
                    np.savetxt(f"{out_base}_shift.csv", T_shift, delimiter=",")
                    
                    # Save Rounded Samples 
                    if SAVE_ROUNDED_SAMPLES:
                        out_rounded_file = os.path.join(DIR_ROUNDED_SAMPLES, f"{base_name}_{method_name}.csv")
                        np.savetxt(out_rounded_file, samples_array, delimiter=",", fmt="%.5f")
                        print(f"     [INFO] Saved {samples_array.shape[0]} rounded samples.")

                    # Save / Verify Transformed Samples 
                    if SAVE_TRANSFORMED_SAMPLES or VERIFY_SAMPLES:
                        try:
                            # Reverse Transform: x_trans = T_matrix * x_rounded + shift
                            X_trans = samples_array @ T_matrix.T + T_shift.flatten()
                            
                            if VERIFY_SAMPLES:
                                violations = (X_trans @ A_trans.T) - b_trans
                                tol = 1e-6
                                valid_samples_mask = np.all(violations <= tol, axis=1)
                                num_valid = np.sum(valid_samples_mask)
                                num_total = X_trans.shape[0]
                                
                                if num_valid == num_total:
                                    print(f"     [VERIFY] SUCCESS: All {num_total} mapped samples satisfy A_trans constraints (tol={tol}).")
                                else:
                                    max_violation = np.max(violations)
                                    print(f"     [VERIFY] WARNING: {num_total - num_valid}/{num_total} samples violate A_trans constraints! Max violation: {max_violation:.4e}")
                            
                            if SAVE_TRANSFORMED_SAMPLES:
                                # Reverse Null-Space Transform to map back to the *simplified* space
                                X_orig_transposed = trans_poly.back_transform(X_trans.T)
                                X_orig = X_orig_transposed.T 
                                
                                out_transformed_file = os.path.join(DIR_TRANSFORMED_SAMPLES, f"{base_name}_{method_name}.csv")
                                np.savetxt(out_transformed_file, X_orig, delimiter=",", fmt="%.5f")
                                print(f"     [INFO] Saved mapped-back samples.")
                        except Exception as e:
                            print(f"     [ERROR] Failed to map back or verify samples: {e}")

                    # Append metrics to global CSV 
                    with open(bench_file_rounding, "a") as f:
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
            print(f"  [ERROR] Failed processing model {base_name}: {e}")

if __name__ == "__main__":
    full_pipeline()