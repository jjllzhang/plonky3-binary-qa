# p3-binary-qa

A systematic quadratic quotient-ring encoder over Plonky3's binary tower fields. The crate supports narrow witnesses, interleaved batches, and row-major matrices. Its Criterion benchmark compares QA encoding with Plonky3's additive FFT Reed–Solomon encoders.

The QA source comes from commit 14278faa8af9980ed9794574c00135b812969edb. The manifest pins the upstream [Plonky3 dependencies](https://github.com/Plonky3/Plonky3/tree/0d803737b681774186cd283b1fd2d8e493655c82). Cargo fetches upstream Plonky3 into its dependency cache and compiles the crates required for the selected build. The additive FFT comparison dependencies are dev-dependencies used by the benchmark.

## Tests and benchmarks

```sh
cargo test --locked
cargo test --locked --features parallel
RUSTFLAGS="-C target-cpu=native" cargo bench --locked --features parallel --bench qa_vs_additive_rs
```

The benchmark includes configurations up to 2²⁴ input elements. To run a smaller case, pass a Criterion filter such as `qa_full/n20_h5_c2` after `--`.
