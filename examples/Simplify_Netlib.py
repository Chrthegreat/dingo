import pandas as pd
import numpy as np
import os
import glob
import time
from PolyRound.mutable_classes.polytope import Polytope
from PolyRound.api import PolyRoundApi
from PolyRound.settings import PolyRoundSettings

# THIS CODE USED THE NULL SPACE .mat EXTRACTS I GOT FROM MATLAB TO SIMPLIFY THEM USING POLYROUND
# FILES ARE IN NetLib_extracts folders and are returned in polyround_output.

# --- CONFIGURATION
# Folder where MATLAB saved the CSVs
input_dir = "netlib_better"  
# Folder where we will save the simplified versions
output_dir = "netlib_simple_better" 

def batch_simplify():
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all "_A.csv" files in the input directory
    a_files = glob.glob(os.path.join(input_dir, "*_A.csv"))
    
    if not a_files:
        print(f"No *_A.csv files found in {input_dir}")
        return

    print(f"Found {len(a_files)} polytopes to process.\n")

    for file_path_A in a_files:
        # Extract the base name (e.g., 'afiro')
        filename_A = os.path.basename(file_path_A)
        base_name = filename_A.replace("_A.csv", "")
        
        # Construct the expected path for b
        file_path_b = os.path.join(input_dir, f"{base_name}_b.csv")
        
        if not os.path.exists(file_path_b):
            print(f"Skipping {base_name}: Found A matrix but missing {base_name}_b.csv")
            continue

        print(f"--- Processing: {base_name} ---")
        try:
            # Load Data
            df_A = pd.read_csv(file_path_A, header=None, sep=',')
            df_b = pd.read_csv(file_path_b, header=None, sep=',')
            A_mat = df_A.values
            b_vec = df_b.values.flatten()

            # Simplify
            poly = Polytope(A_mat, b_vec)
            settings = PolyRoundSettings()
            settings.thresh = 1e-6
            settings.hp_flags['FeasibilityTol'] = 1e-6
            settings.hp_flags['OptimalityTol']  = 1e-6   

            # Print current settings for verification
            print("PolyRound settings in use:")
            for key, value in settings.__dict__.items():
                print(f"  {key}: {value}")
            
            start_t = time.time()
            simplified_poly = PolyRoundApi.simplify_polytope(poly, settings, normalize=True)
            elapsed = time.time() - start_t
            
            # Report Stats
            orig_cons = A_mat.shape[0]
            final_cons = simplified_poly.A.shape[0]
            reduction = 100 * (orig_cons - final_cons) / orig_cons
            
            print(f"Simplification done in {elapsed:.2f}s")
            print(f"Constraints: {orig_cons} -> {final_cons} ({reduction:.1f}% reduction)")

            # Save Results
            out_A = os.path.join(output_dir, f"{base_name}_A_simple.csv")
            out_b = os.path.join(output_dir, f"{base_name}_b_simple.csv")
            
            pd.DataFrame(simplified_poly.A).to_csv(out_A, index=False, header=False)
            pd.DataFrame(simplified_poly.b).to_csv(out_b, index=False, header=False)
            
            if simplified_poly.S is not None and simplified_poly.S.size > 0:
                out_S = os.path.join(output_dir, f"{base_name}_S_simple.csv")
                out_h = os.path.join(output_dir, f"{base_name}_h_simple.csv")
                pd.DataFrame(simplified_poly.S).to_csv(out_S, index=False, header=False)
                pd.DataFrame(simplified_poly.h).to_csv(out_h, index=False, header=False)
                print(f"Hidden equalities extracted and saved.")

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
        
        print("") # Empty line for readability

if __name__ == "__main__":
    batch_simplify()