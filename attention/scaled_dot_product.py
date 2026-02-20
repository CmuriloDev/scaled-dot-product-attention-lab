import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
   
    x = np.asarray(x, dtype=np.float64) 
    x_max = np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / denom

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    *,
    return_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
   

    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K and V must be 2D matrices (rank-2).")

    n_q, d_k_q = Q.shape
    n_k, d_k_k = K.shape
    n_k_v, d_v = V.shape

    if d_k_q != d_k_k:
        raise ValueError(f"Dimension mismatch d_k: Q={d_k_q}, K={d_k_k}.")
    if n_k != n_k_v:
        raise ValueError(f"Incompatibility of n_k: K has {n_k} lines, V has {n_k_v}.")

    d_k = d_k_q
    if d_k <= 0:
        raise ValueError("d_k must be positive.")

    # 1) Dot-product scores: (n_q, n_k)
    scores = Q @ K.T

    # 2) Scaling factor: divide by sqrt(d_k)
    scores = scores / np.sqrt(d_k)

    # 3) Row-wise softmax => attention weights: (n_q, n_k)
    weights = softmax(scores, axis=1)

    # 4) Weighted sum of V: (n_q, d_v)
    output = weights @ V

    return (output, weights) if return_weights else output

if __name__ == "__main__":
    Q = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    V = np.array([[10.0, 0.0],
                  [0.0, 20.0]])

    out, w = scaled_dot_product_attention(Q, K, V)
    np.set_printoptions(precision=6, suppress=True)
    print("Attention weights:\n", w)
    print("Output:\n", out)