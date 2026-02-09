# VectorAdd Benchmark Report

## Purpose
Baseline benchmark for the first CUDA kernel (`vector_add`).

## Reproduce
```bash
cd learning/01-fundamentals/vector_add
make
./vector_add           # default n=2^24
./vector_add 10000000  # custom size
```

## Metrics
- Correctness: `max_abs_err`
- CPU time (avg ms)
- GPU kernel time (avg ms)
- Effective bandwidth (GB/s)

## Notes Template
- GPU model:
- CUDA version:
- n:
- block size:
- cpu_ms:
- gpu_kernel_ms:
- speedup (cpu/gpu):
- effective_bw(GB/s):
- observation:
