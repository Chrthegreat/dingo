import os
import glob
import time
import pandas as pd
from PolyRound.api import PolyRoundApi
from PolyRound.settings import PolyRoundSettings

# THIS SCRIPT I MADE TO SIMPLIFY THE .XML BIOLOGICAL POLYTOPES I DOWNLOADED FROM BIGG.

# --- CONFIGURATION ---
input_dir = "models_biology" 
output_dir = "models_simplified" 

def batch_process_sbml():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all XML files
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    
    if not xml_files:
        print(f"No .xml files found in {input_dir}")
        return

    print(f"Found {len(xml_files)} biological models to process.\n")

    results = []

    for file_path in xml_files:
        base_name = os.path.basename(file_path).replace(".xml", "")
        print(f"--- Processing: {base_name} ---")

        try:
            # Load SBML directly into Polytope
            print("  Loading SBML model...")
            poly = PolyRoundApi.sbml_to_polytope(file_path)
            
            # Set up Polyround with some relaxed settings
            settings = PolyRoundSettings()
            # settings.numerics_threshold = 1e-6
            # settings.hp_flags['FeasibilityTol'] = 1e-6
            # settings.hp_flags['OptimalityTol']  = 1e-6
            # settings.verbose = False

            # Simplify
            print("  Simplifying polytope...")
            start_t = time.time()
            simplified_poly = PolyRoundApi.simplify_polytope(poly, settings)
            elapsed = time.time() - start_t
            
            # Report Stats
            orig_cons = poly.A.shape[0] if poly.A is not None else 0
            final_cons = simplified_poly.A.shape[0]
            reduction = 100 * (orig_cons - final_cons) / orig_cons if orig_cons > 0 else 0
            print(f"  Done in {elapsed:.2f}s! Constraints: {orig_cons} -> {final_cons} ({reduction:.1f}% reduction)")

            results.append({
                "Model": base_name,
                "Time (s)": round(elapsed, 2),
                "Initial Constraints": orig_cons,
                "Final Constraints": final_cons,
                "% Reduction": round(reduction, 1),
            })

            # Save Results to CSV
            out_A = os.path.join(output_dir, f"{base_name}_A.csv")
            out_b = os.path.join(output_dir, f"{base_name}_b.csv")
            pd.DataFrame(simplified_poly.A).to_csv(out_A, index=False, header=False)
            pd.DataFrame(simplified_poly.b).to_csv(out_b, index=False, header=False)
            
            # Handle equalities if any exist in the simplified model
            if simplified_poly.S is not None and simplified_poly.S.size > 0:
                out_S = os.path.join(output_dir, f"{base_name}_S.csv")
                out_h = os.path.join(output_dir, f"{base_name}_h.csv")
                pd.DataFrame(simplified_poly.S).to_csv(out_S, index=False, header=False)
                pd.DataFrame(simplified_poly.h).to_csv(out_h, index=False, header=False)
                print("  Saved equality constraints (S, h).")

        except Exception as e:
            print(f"  Error processing {base_name}: {e}")
        
        print("") # Added empty line for readability

    if results:
        summary_path = os.path.join(output_dir, "biology_simplify.csv")
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        
if __name__ == "__main__":
    batch_process_sbml()