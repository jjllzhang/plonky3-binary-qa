# Binary QA codes

This project studies systematic quadratic quotient-ring codes over binary extension fields. We compare their encoding cost with additive FFT Reed-Solomon encoding across different input fields and batch layouts.

## The code

Let $F$ be a binary extension field containing $\mathbb{F}_4$. A message of length $n=2^s$ is the coefficient vector of a multilinear polynomial $v$ in the quotient ring

$$
R_s = F[X_1,\ldots,X_s] \,/\,
\bigl(X_1^2+X_1+1,\ldots,X_s^2+X_s+1\bigr).
$$

Given fixed public ring elements $a_1,\ldots,a_{c-1}$, the codeword is $(v,a_1v,\ldots,a_{c-1}v)$. The message appears verbatim as the first block, and the rate is $1/c$. The benchmark uses rates $1/2$ and $1/4$.

The roots $\omega$ and $\omega^2$ of $X^2+X+1$ give two evaluation points on each axis. The Chinese remainder theorem identifies $R_s$ with $F^n$ by evaluating on $\{\omega,\omega^2\}^s$. Encoding evaluates the message once, multiplies its evaluations by the precomputed evaluations of each $a_i$, and interpolates each parity block. Each butterfly uses two additions and one multiplication by $\omega$. With the multiplier evaluations fixed, encoding takes $O(n\log n)$ field operations per message at a fixed rate.

## What we measure

When message coefficients lie in a subfield containing $\mathbb{F}_4$, the entire forward transform can stay in that subfield. If the parity multipliers lie in a larger field, computing the parity blocks first needs that field at the pointwise product. Additive FFT Reed-Solomon encoders also use small subfield constants. We measure how input width affects the cost of complete encodings.

The benchmark covers interleaved batches, narrow inputs with parity in $\mathbb{F}_{2^{128}}$, and row-major square matrices. It matches message lengths and rates. The square-matrix comparison includes the input and output transposes required by the Reed-Solomon path.

## Reproduce

The comparison uses a [fixed Plonky3 revision](https://github.com/Plonky3/Plonky3/tree/0d803737b681774186cd283b1fd2d8e493655c82).

```sh
cargo test --locked --features parallel
RUSTFLAGS="-C target-cpu=native" cargo bench --locked --features parallel --bench qa_vs_additive_rs -- n20_h5_c2
```

Omit the final filter to run the full comparison.
