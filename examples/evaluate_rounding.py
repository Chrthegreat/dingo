import numpy as np
import time
from dingo import PolytopeSampler
from scipy.linalg import eigh

def _ess_min(samples):
    """Minimum effective sample size across dimensions using Geyer's initial sequence estimator.

    samples: ndarray of shape (n_dims, n_samples)  (as returned by dingo samplers)
    Returns the minimum ESS over all dimensions.
    """
    # samples shape: (d, N) – transpose to (N, d)
    X = samples.T
    N, d = X.shape
    ess_vals = []
    for j in range(d):
        x = X[:, j] - X[:, j].mean()
        # Normalised autocorrelation via FFT
        n = len(x)
        fft_size = 1
        while fft_size < 2 * n:
            fft_size <<= 1
        f = np.fft.rfft(x, n=fft_size)
        acf_full = np.fft.irfft(f * np.conj(f))[:n]
        acf = acf_full / acf_full[0]   # normalise so lag-0 = 1
        # Initial positive sequence: sum pairs (lag 2k, 2k+1) while positive
        rho_sum = 0.0
        for k in range(1, n // 2):
            pair = acf[2 * k - 1] + acf[2 * k]
            if pair < 0:
                break
            rho_sum += pair
        eff_n = N / (1.0 + 2.0 * rho_sum)
        ess_vals.append(min(eff_n, N))
    return min(ess_vals)

def evaluate_rounding_quality(A, b, T_matrix=None, walk_method='billiard_walk', n_samples=None, verbose=True):
    """Compute quality metrics for a rounded polytope {x : A x <= b}."""
    
    d = A.shape[1]
    if n_samples is None:
        n_samples = max(20,40000)

    burn_in  = int(5 * np.sqrt(d))
    thinning = max(1, burn_in)

    burn_in  = 0
    thinning = 1

    if verbose:
        print(f"     [quality] sampling {n_samples} points in R^{d} (burn={burn_in}, thin={thinning}) ...")

    start_sampling_time = time.time()
    samples = PolytopeSampler.sample_from_polytope_no_multiphase(
        A, b,
        method=walk_method,
        n=n_samples,
        burn_in=burn_in,
        thinning=thinning
    )
    sampling_elapsed_time = time.time() - start_sampling_time

    # Condition number T, how hard,big was the rounding
    if T_matrix is not None:
        sv = np.linalg.svd(T_matrix, compute_uv=False)
        #sv = sv[sv > 1e-12]
        t_cond = sv.max() / sv.min() if sv.size > 0 else float('inf')
    else:
        t_cond = None

    # Covariance of the samples
    X = samples.T                                  
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    
    eigvals = eigh(cov, eigvals_only=True) 
    cov_ratio = eigvals[-1] / eigvals[0] if eigvals[0] > 1e-30 else float('inf')
    min_eigenvalue = eigvals[0]
    
    # Trace
    cov_trace = np.trace(cov)
    
    # Diagonal stats
    diag_elements = np.diag(cov)
    diag_min = np.min(diag_elements)
    diag_max = np.max(diag_elements)
    diag_mean = np.mean(diag_elements)
    diag_std = np.std(diag_elements)
  
    # Distance to Identity Matrix
    I = np.eye(d)
    cov_diff = cov - I
    dist_to_identity_fro = np.linalg.norm(cov_diff, 'fro') # Frobenius norm (overall distance)
    dist_to_identity_mean = np.mean(np.abs(cov_diff))      # Average element-wise error
    
    mask = ~np.eye(d, dtype=bool) # This will return a matrix will False in the diagonal and true elsewhere.
    non_diagonals = cov[mask] # Now, will keep only matrix values corresponding to true, thus the non diagonal
    
    if non_diagonals.size > 0:
        non_diag_min = np.min(non_diagonals)
        non_diag_max = np.max(non_diagonals)
        non_diag_mean = np.mean(non_diagonals)
        non_diag_std = np.std(non_diagonals)
    else:
        non_diag_min = non_diag_max = non_diag_std = 0.0

    # Minimum ESS 
    ess = _ess_min(samples) 

    metrics = {
        "T_cond": t_cond,
        "cov_ratio": cov_ratio,
        "ess_min": ess,
        "cov_trace": cov_trace,
        "diag_min": diag_min,
        "diag_max": diag_max,
        "diag_mean": diag_mean,
        "diag_std": diag_std,
        "dist_to_I_fro": dist_to_identity_fro,
        "dist_to_I_mean": dist_to_identity_mean,
        "non_diag_min": non_diag_min,
        "non_diag_max": non_diag_max,
        "non_diag_mean": non_diag_mean,
        "non_diag_std": non_diag_std,
        "sampling_time": sampling_elapsed_time,
        "min_eigenvalue": min_eigenvalue,
    }

    return metrics, X