import numpy as np
from attention.scaled_dot_product import scaled_dot_product_attention


def test_shapes_and_row_softmax_property():
    # 2 queries, 3 keys, d_k=4, d_v=2
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(2, 4))
    K = rng.normal(size=(3, 4))
    V = rng.normal(size=(3, 2))

    out, w = scaled_dot_product_attention(Q, K, V)

    assert w.shape == (2, 3)
    assert out.shape == (2, 2)

    # softmax per row => each row sums to 1
    assert np.allclose(w.sum(axis=1), np.ones(2), atol=1e-9)

def test_prefers_more_aligned_key():
    # Query aligns with first key more than the second key
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    V = np.array([[5.0, 0.0],
                  [0.0, 100.0]])

    out, w = scaled_dot_product_attention(Q, K, V)

    assert w[0, 0] > w[0, 1]
    # output should be closer to the first value vector
    assert out[0, 0] > out[0, 1]