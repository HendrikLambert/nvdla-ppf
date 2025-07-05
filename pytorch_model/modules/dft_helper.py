import torch
from mpmath import mp
from functools import lru_cache

@lru_cache(maxsize=10)
def create_dft_matrix(n: int) -> torch.Tensor:
    """
    Creates the (2n x 2n) real-valued matrix that performs the n-point DFT
    on interleaved real/imaginary inputs.
    
    Uses mpmath for high precision calculations of the twiddle factors.
    Uses a cache to avoid recomputing the matrix for the same n.
    
    Args:
        n (int): The number of complex DFT points.
    """
    
    matrix = torch.zeros((2 * n, 2 * n), dtype=torch.float64)
    # Set precision for mpmath
    mp.dps = 64

    for k in range(n):  # Index for the output DFT coefficient X_k
        for j in range(n):  # Index for the input sample x_j
            # Twiddle factor W_n^{kj} = cos(2*pi*k*j/n) - i*sin(2*pi*k*j/n)
            angle = 2 * mp.pi * k * j / n
            re_w = float(mp.cos(angle))
            im_w = -float(mp.sin(angle))

            # Contribution of x_j = Re(x_j) + i*Im(x_j) to X_k = Re(X_k) + i*Im(X_k)
            # X_k = sum_j (Re(x_j) + i*Im(x_j)) * (re_w + i*im_w)
            #     = sum_j [ (Re(x_j)*re_w - Im(x_j)*im_w) + i*(Re(x_j)*im_w + Im(x_j)*re_w) ]

            # Row for Re(X_k) is 2*k
            # Col for Re(x_j) is 2*j
            # Col for Im(x_j) is 2*j + 1

            # Re(x_j) * Re(W)
            matrix[2 * k, 2 * j] = re_w
            # -Im(x_j) * Im(W) (since Im(W) for X_k is positive: Im(x_j)*(-Im(W_twiddle)))
            matrix[2 * k, 2 * j + 1] = -im_w
            # or easier, Re(x_j)*re_w - Im(x_j)*im_w.
            # The weight for Im(x_j) is -im_w.

            # Row for Im(X_k) is 2*k + 1
            matrix[2 * k + 1, 2 * j] = im_w  # Re(x_j) * Im(W)
            matrix[2 * k + 1, 2 * j + 1] = re_w  # Im(x_j) * Re(W)

    return matrix.to(torch.float32)
