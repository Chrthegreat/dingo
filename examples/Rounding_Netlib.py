import numpy as np
import pandas as pd
from dingo import PolytopeSampler
import sys
import os
import time

from evaluate_rounding import evaluate_rounding_quality

# ==============================================================================
# 
# WHAT THIS CODE DOES:
# 1. Scans the 'INPUT FOLDER' folder for simplified polytope files 
#    (matrices A and b from the previous step).
# 2. Loops through a list of problem names (e.g., 'afiro', 'blend').
# 3. Applies specific Rounding Methods (e.g., Min Volume Ellipsoid) using 
#    the 'dingo' library. This transforms the polytope into a shape that is 
#    easier to sample from (improves the condition number).
# 4. Measures the time taken for each rounding operation.
# 5. Exports the new, rounded matrices to the 'OUTPUT FOLDER' folder.
#
# =============================================================================

#INPUT_DIR = "polyround_output"   # Where the simplified files are
INPUT_DIR = "netlib_no_normalize"
OUTPUT_DIR = "rounded_no_normal_output"    # Where the rounded files will go
BENCH_FILE = "Netlib_benchmarks.csv"
PRINT_CONSOLE_RESULTS = False

# Only include the base names (no _A, _simple, or extension)
PROBLEMS = [
    #"afiro",
    #"blend",
    #"beaconfd",  
    #"scorpion",  
    #"agg",      
    "etamacro"
    #"degen2"
    # "sierra",  # Skipped (Too large for simplification)
    # "degen3",  # Skipped
    # "25fv47"   # Skipped
]

# ROUNDING METHODS
# Format: ("method_name", Active_Boolean)
METHODS = [
    ("min_ellipsoid", False),   
    ("john_position", False), 
    ("isotropic_position", False),    
    ("log_barrier", True),       
    ("vaidya_barrier", True),    
    ("volumetric_barrier", True) 
]

def round_and_export_netlib(problem_name, method_name):
    print(f"\n{'='*60}")
    print(f"Processing: {problem_name} | Method: {method_name}")
    
    # Construct File Paths (Looking for SIMPLIFIED files)
    # File pattern: {name}_A_simple.csv inside polyround_output/
    A_file = os.path.join(INPUT_DIR, f"{problem_name}_A_simple.csv")
    b_file = os.path.join(INPUT_DIR, f"{problem_name}_b_simple.csv")

    # Check existence
    if not os.path.exists(A_file) or not os.path.exists(b_file):
        print(f"Skipping: Could not find files in {INPUT_DIR}")
        print(f"Expected: {A_file}")
        return

    # Load the matrices
    print("  Reading simplified CSV files...")
    try:
        # Force sep=',' and contiguous arrays for C-engine speed
        A = pd.read_csv(A_file, header=None, sep=',').values.astype(np.float64, order='C')
        b = pd.read_csv(b_file, header=None, sep=',').values.flatten().astype(np.float64, order='C')
        
        A = np.ascontiguousarray(A)
        b = np.ascontiguousarray(b)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Dimension: {A.shape[1]}")
    print(f"Constraints: {A.shape[0]}")
    print(f"Running {method_name}...")
    
    start_time = time.time()
    success = False
    
    try:
        A_r, b_r, T, T_shift = PolytopeSampler.round_polytope(
            A, b, method=method_name
        )
        success = True
    except Exception as e:
        print(f"Rounding failed: {e}")

    # Used to avoid crashing if eigen fails
    # A_r, b_r, T, T_shift, success, err = safe_round(A, b, method_name)

    # if not success:
    #     print(f"Rounding failed for {problem_name} | {method_name}: {err}")
    #     return
        
    # Stop Timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if success:
        print(f"Success!")
        print(f"Time taken: {elapsed_time:.4f} seconds")

        print("Evaluating rounding quality...")

        metrics = evaluate_rounding_quality(
            A_r,
            b_r,
            T_matrix=T,
            verbose=False
        )

        # Diagnostics
        if PRINT_CONSOLE_RESULTS:
            print(f"T condition number      : {metrics['T_cond']:.4g}")
            print(f"Covariance ratio        : {metrics['cov_ratio']:.4g}")
            print(f"ESS min                 : {metrics['ess_min']:.1f}")
            print(f"Covariance trace        : {metrics['cov_trace']:.4g}")

            print(
                f"Diag elements           : "
                f"Mean={metrics['diag_mean']:.4g}, "
                f"Min={metrics['diag_min']:.4g}, "
                f"Max={metrics['diag_max']:.4g}, "
                f"Std={metrics['diag_std']:.4g}"
            )

            print(f"Distance to Identity    : {metrics['dist_to_I_fro']:.4g}")
            print(f"Non-diagonal std        : {metrics['non_diag_std']:.4e}")

        with open(BENCH_FILE, "a") as f:
            f.write(
                f"{problem_name},{method_name},{A.shape[1]},{elapsed_time:.4f},"
                f"{metrics['T_cond']:.4e},"
                f"{metrics['cov_ratio']:.4e},"
                f"{metrics['ess_min']:.4f},"
                f"{metrics['cov_trace']:.4e},"
                f"{metrics['diag_mean']:.4e},"
                f"{metrics['diag_min']:.4e},"
                f"{metrics['diag_max']:.4e},"
                f"{metrics['diag_std']:.4e},"
                f"{metrics['dist_to_I_fro']:.4e},"
                f"{metrics['non_diag_std']:.4e}\n"
            )

        # We save to the OUTPUT_DIR
        out_name_base = f"{problem_name}_{method_name}"
        output_A = os.path.join(OUTPUT_DIR, f"{out_name_base}_A.csv")
        output_b = os.path.join(OUTPUT_DIR, f"{out_name_base}_b.csv")
        
        print(f"  Saving to {OUTPUT_DIR}/...")
        np.savetxt(output_A, A_r, delimiter=",")
        np.savetxt(output_b, b_r, delimiter=",")

if __name__ == "__main__":
   
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    if not os.path.exists(BENCH_FILE):
        with open(BENCH_FILE, "w") as f:
            f.write(
                "Problem,Method,Dimension,Time_sec,"
                "T_cond,Cov_ratio,ESS_min,Cov_trace,"
                "Diag_mean,Diag_min,Diag_max,Diag_std,"
                "Dist_I,NonDiag_std\n"
            )

    print(f"Starting batch rounding for {len(PROBLEMS)} problems...")
    print(f"Input Directory: {INPUT_DIR}")
    
    # Outer Loop: Iterate through the list of problems
    for problem in PROBLEMS:
        
        # Inner Loop: Iterate through the list of methods
        for method_name, is_enabled in METHODS:
            if is_enabled:
                round_and_export_netlib(problem, method_name)
                
    print(f"\n{'='*60}")
    print("Batch processing complete.")
