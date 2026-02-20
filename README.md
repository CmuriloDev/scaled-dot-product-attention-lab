# LAB P1-01 --- Scaled Dot-Product Attention (Self-Attention)

This repository implements the core mechanism of **Scaled Dot-Product
Attention**, following the formula from the paper *Attention Is All You
Need*.

The implementation follows the lab requirement: **no high-level Deep
Learning layers** (e.g., `nn.MultiheadAttention`), using only **NumPy**
for linear algebra.

------------------------------------------------------------------------

## Reference Formula

Attention(Q, K, V) = softmax((QK\^T) / sqrt(d_k)) V

**Important:** softmax is applied **row-wise** over the score matrix.
Each query produces a probability distribution over all keys.

------------------------------------------------------------------------

## Why divide by sqrt(d_k)?

The dot product QK\^T grows in magnitude as d_k increases.\
Without normalization, the softmax function may saturate (values become
extremely close to 0 or 1), leading to unstable gradients and poor
numerical behavior.

Dividing by sqrt(d_k) keeps values in a controlled range, improving
stability.

------------------------------------------------------------------------

## Project Structure

-   attention/scaled_dot_product.py → Core implementation\
-   tests/test_attention.py → Unit tests\
-   requirements.txt → Dependencies

------------------------------------------------------------------------

## How to Run

### 1) Create virtual environment

python -m venv .venv

Activate:

Windows: .venv`\Scripts`{=tex}`\activate`{=tex}

Linux/macOS: source .venv/bin/activate

### 2) Install dependencies

pip install -r requirements.txt

### 3) Run tests

pytest -q

### 4) Run demo

python -m attention.scaled_dot_product

------------------------------------------------------------------------

## Example

Input:

Q = \[\[1,0\], \[0,1\]\]

K = \[\[1,0\], \[0,1\]\]

V = \[\[10,0\], \[0,20\]\]

After computation:

weights ≈ \[\[0.669761, 0.330239\], \[0.330239, 0.669761\]\]

output ≈ \[\[ 6.69761, 6.60478\], \[ 3.30239, 13.39522\]\]

------------------------------------------------------------------------

License: Educational Use
