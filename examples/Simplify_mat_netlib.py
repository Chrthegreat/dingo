import os
import glob
import time
import scipy.io as sio
import scipy.sparse as sp
import pandas as pd
import numpy as np
from PolyRound.api import PolyRoundApi
from PolyRound.mutable_classes.polytope import Polytope
from PolyRound.settings import PolyRoundSettings

# --- CONFIGURATION ---
INPUT_DIR = "netlib_mat"
OUTPUT_DIR = "netlib_mat_output"

ALLOWED_MODELS = [
    # "lp_afiro",
    # "lpi_bgetam",
    # "lp_adlittle",
    # "lp_etamacro",
    # "lp_scorpion",
    # "lp_sierra",
    # "lp_degen2",
    # "lp_cycle",
    # "lp_degen3",
    # "lp_stocfor1",
    # "lp_stocfor2",
    # "lp_beaconfd",
    # "lp_bore3d",
    # "lp_grow22",
    # "lp_blend",
    # #"lp_80bau3b"
    # "lp_25fv47",
     "lp_agg",
    # "lp_agg2",
     "lp_bandm",
    # "lp_bnl2",
    # "lp_brandy",
      "lp_carpi",
    # "lp_grow22",
     "lpi_bgdbg1",
     "lpi_bgetam",
     "lpi_box1",
     "lpi_cplex2"
]
# Upper and lower bouund arte inside problem/aux
def extract_lp_data(problem, num_vars):
    """Safely extracts bounds, objective function (c), and offset (z0)"""
    lb = np.zeros(num_vars)
    ub = np.full(num_vars, np.inf)
    c = np.zeros(num_vars)
    z0 = 0.0

    if 'lb' in problem.dtype.names:
        lb = problem['lb'].flatten()
    if 'ub' in problem.dtype.names:
        ub = problem['ub'].flatten()
    if 'c' in problem.dtype.names:
        c = problem['c'].flatten()

    if 'aux' in problem.dtype.names:
        aux = problem['aux'][0, 0]
        if 'lo' in aux.dtype.names:
            lb = aux['lo'].astype(float).flatten()
        elif 'lb' in aux.dtype.names:
            lb = aux['lo'].astype(float).flatten()
            
        if 'hi' in aux.dtype.names:
            ub = aux['hi'].astype(float).flatten()
        elif 'ub' in aux.dtype.names:
            ub = aux['hi'].astype(float).flatten()
            
        if 'c' in aux.dtype.names:
            c = aux['c'].flatten()
        if 'z0' in aux.dtype.names:
            z0 = float(aux['z0'][0, 0]) 

    return lb, ub, c, z0

def batch_process_netlib():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    summary_path = "netlb_simplify.csv"
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write("Model,Simplify Time (s),Transform Time (s),Total Time (s),Initial Dimension,Final Dimension,Dim Reduction %,Initial Inequalities,Final Inequalities,Ineq Reduction %\n")
        print(f"--- Created new benchmark file: {summary_path} ---")

    mat_files = glob.glob(os.path.join(INPUT_DIR, "*.mat"))
    
    if not mat_files:
        print(f"No .mat files found in {INPUT_DIR}")
        return

    # Filter files based on ALLOWED_MODELS if it is defined
    if ALLOWED_MODELS is not None:
        mat_files = [
            f for f in mat_files 
            if os.path.basename(f).replace(".mat", "") in ALLOWED_MODELS
        ]
        
    if not mat_files:
        print("No models matched the ALLOWED_MODELS filter.")
        return

    print(f"Found {len(mat_files)} Netlib models to process.\n")

    for file_path in mat_files:
        base_name = os.path.basename(file_path).replace(".mat", "")
        print(f"--- Processing: {base_name} ---")

        try:
            # LOAD MAT FILE
            mat_data = sio.loadmat(file_path)
            struct_name = 'Problem' if 'Problem' in mat_data else 'problem'
            problem = mat_data[struct_name][0, 0]
            
            A_raw = problem['A']
            if hasattr(A_raw, 'toarray'):
                A_raw = A_raw.toarray()
                
            b_raw = problem['b'].flatten()
            num_constraints, num_vars = A_raw.shape
            
            lb_raw, ub_raw, c_raw, z0_raw = extract_lp_data(problem, num_vars)

            # I will replace infinite bounds with artificial finite bounds
            BIG = 1e12

            lb_raw = lb_raw.astype(float)
            ub_raw = ub_raw.astype(float)

            lb_raw[np.isneginf(lb_raw)] = -BIG
            ub_raw[np.isposinf(ub_raw)] = BIG

            var_names = [f"x{i}" for i in range(num_vars)]
            
            # 1) Equalities (S x = h)
            eq_names = [f"eq_c{i}" for i in range(num_constraints)]
            S_df = pd.DataFrame(A_raw, index=eq_names, columns=var_names)
            h_series = pd.Series(b_raw, index=eq_names)
            
            # 2) Inequalities (A x <= b)
            I = sp.eye(num_vars, format='csc')
            A_bound = sp.vstack([I, -I], format='csc')
            b_bound = np.concatenate([ub_raw, -lb_raw])

            A_bound_valid = A_bound.toarray()
            b_bound_valid = b_bound
                        
            ineq_names = [f"ineq_c{i}" for i in range(len(b_bound_valid))]
            A_df = pd.DataFrame(A_bound_valid, index=ineq_names, columns=var_names)
            b_series = pd.Series(b_bound_valid, index=ineq_names)

            # Instantiate Polytope
            poly = Polytope(A=A_df, b=b_series, S=S_df, h=h_series)

            settings = PolyRoundSettings()
            # settings.numerics_threshold = 1e-5
            # settings.hp_flags['FeasibilityTol'] = 1e-5
            # settings.hp_flags['OptimalityTol']  = 1e-5

            orig_ineq = poly.A.shape[0] if poly.A is not None else 0
            orig_vars = num_vars

            # ******SIMPLIFY******
            print("  Simplifying polytope...")
            start_t = time.time()
            simplified_poly = PolyRoundApi.simplify_polytope(poly, settings, normalize=True)
            simplify_time = time.time() - start_t

            # ******TRANSFORM*******
            print("  Transforming polytope (Null Space Projection)...")
            start_t2 = time.time()
            transformed_poly = PolyRoundApi.transform_polytope(simplified_poly, settings)
            transform_time = time.time() - start_t2
            
            total_time = simplify_time + transform_time

            # Metrics
            final_ineq = transformed_poly.A.shape[0] if transformed_poly.A is not None else 0
            final_vars = transformed_poly.A.shape[1] if transformed_poly.A is not None else 0
            
            ineq_reduction = 100 * (orig_ineq - final_ineq) / orig_ineq if orig_ineq > 0 else 0
            dim_reduction = 100 * (orig_vars - final_vars) / orig_vars if orig_vars > 0 else 0

            print(f"  Done in {total_time:.2f}s!")
            print(f"  Dimension (Vars): {orig_vars} -> {final_vars} ({dim_reduction:.1f}% reduction)")
            print(f"  Inequalities:     {orig_ineq} -> {final_ineq} ({ineq_reduction:.1f}% reduction)")

            result_row = {
                "Model": base_name,
                "Simplify Time (s)": round(simplify_time, 2),
                "Transform Time (s)": round(transform_time, 2),
                "Total Time (s)": round(total_time, 2),
                "Initial Dimension": orig_vars,
                "Final Dimension": final_vars,
                "Dim Reduction %": round(dim_reduction, 1),
                "Initial Inequalities": orig_ineq,
                "Final Inequalities": final_ineq,
                "Ineq Reduction %": round(ineq_reduction, 1),
            }

            out_A = os.path.join(OUTPUT_DIR, f"{base_name}_A.csv")
            out_b = os.path.join(OUTPUT_DIR, f"{base_name}_b.csv")

            A_final = transformed_poly.A.values if hasattr(transformed_poly.A, 'values') else transformed_poly.A
            b_final = transformed_poly.b.values.flatten() if hasattr(transformed_poly.b, 'values') else transformed_poly.b.flatten()

            pd.DataFrame(A_final).to_csv(out_A, index=False, header=False)
            pd.DataFrame(b_final).to_csv(out_b, index=False, header=False)

            pd.DataFrame([result_row]).to_csv(summary_path, mode='a', index=False, header=False)
            print(f"  -> Result saved to {summary_path}")

        except Exception as e:
            print(f"  [!] Error processing {base_name}: {e}")
        
        print("")

    print("--- Batch processing complete! ---")

if __name__ == "__main__":
    batch_process_netlib()