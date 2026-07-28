import os
import time
import numpy as np
import pandas as pd

from dingo import PolytopeSampler

from PolyRound.api import PolyRoundApi
from PolyRound.mutable_classes.polytope import Polytope as PRPolytope
from PolyRound.settings import PolyRoundSettings

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
    # Benchmark file
    bench_file = "skinny_benchmark_times.csv"
    if not os.path.exists(bench_file):
        with open(bench_file, "w") as f:
            f.write("Model,Method,Dimension,Time_sec\n")
            
    dims_to_test = [500, 1000, 1500, 2000, 2500, 3000]
    skinny_percentages = [0.10, 0.50]
    
    methods = [
        "log_barrier", 
        "vaidya_barrier", 
        "volumetric_barrier",
        "john_position", 
        #"min_ellipsoid",
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
                            
                        else:
                            start_r = time.time()
                            A_r, b_r, T_matrix, T_shift = PolytopeSampler.round_polytope(
                                A, b, method=method_name
                            )
                            elapsed_r = time.time() - start_r
                        
                        print(f"     [SUCCESS] Rounded with {method_name} in {elapsed_r:.4f}s.")

                        # Append timing to global CSV 
                        with open(bench_file, "a") as f:
                            f.write(f"{model_name},{method_name},{A.shape[1]},{elapsed_r:.4f}\n")

                    except Exception as e:
                        print(f"     [FAILED] {method_name}: {e}")
                        continue 

            except Exception as e:
                print(f"  [ERROR] Failed on {base_name}: {e}")

if __name__ == "__main__":
    run_benchmark()
