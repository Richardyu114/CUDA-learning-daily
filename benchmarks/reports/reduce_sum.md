# ReduceSum Benchmark Report

## Reproduce
```bash
cd learning/01-fundamentals/reduce_sum
make
./reduce_sum
./reduce_sum_opt
bash ../../../../scripts/compare_reduce.sh
```

## Metrics
- abs_err
- cpu_ms(avg)
- gpu_ms(avg)
- speedup (baseline / optimized)

## Optimization Used (v1)
- Two-elements-per-thread loading
- Warp-level final reduction (`__shfl_down_sync`)
- Fewer reduction stages in shared memory
