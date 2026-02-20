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

def test_numeric_example_matches_expected():
    # Simple, hand-checkable example
    Q = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    V = np.array([[10.0, 0.0],
                  [0.0, 20.0]])

    out, w = scaled_dot_product_attention(Q, K, V)

    # scores = [[1,0],[0,1]] / sqrt(2)
    s = 1 / np.sqrt(2)
    # softmax([s,0]) and softmax([0,s]) are symmetric
    a = np.exp(s) / (np.exp(s) + 1.0)
    b = 1.0 / (np.exp(s) + 1.0)

    expected_w = np.array([[a, b],
                           [b, a]])
    expected_out = expected_w @ V

    assert np.allclose(w, expected_w, atol=1e-9)
    assert np.allclose(out, expected_out, atol=1e-9)