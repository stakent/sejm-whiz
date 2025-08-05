# Performance Benchmark Suite

This directory contains comprehensive performance analysis and benchmarking tools for the Sejm-Whiz system.

## Directory Structure

```
performance/
├── benchmarks/          # Benchmark scripts and test suites
├── reports/             # Performance analysis reports and documentation
├── results/             # Raw benchmark results and data files
├── scripts/             # Utility scripts for performance testing
└── README.md           # This file
```

## Benchmark Suite Components

### Benchmarks (`benchmarks/`)

**Core Benchmark Scripts:**

- `benchmark_embeddings.py` - Isolated embedding generation performance testing
- `benchmark_full_pipeline.py` - Comprehensive end-to-end pipeline benchmarking
- `benchmark_pipeline_simple.py` - Simplified production-focused benchmark

### Reports (`reports/`)

**Analysis Documentation:**

- `EMBEDDING_PERFORMANCE_BENCHMARK.md` - Complete performance analysis with:
  - GPU vs CPU comparative results
  - Cloud provider cost analysis (7 major providers)
  - Deployment strategy recommendations
  - Implementation roadmaps
  - Multi-cloud deployment planning

### Results (`results/`)

**Raw Benchmark Data:**

- `full_pipeline_benchmark_*.json` - Complete benchmark result datasets
- Performance metrics and timing data
- Component-level performance breakdowns

## Key Findings Summary

**Performance Results:**

- **CPU outperforms GPU by 1.49x** in full pipeline processing
- **Overall throughput**: 3.92 texts/sec (CPU) vs 2.64 texts/sec (GPU)
- **Cost savings**: 58-96% with CPU-only deployments
- **Recommended deployment**: CPU-only Kubernetes clusters

**Deployment Recommendations:**

- **Startup/MVP**: Hetzner Cloud ($89-289/month)
- **Production**: Google Cloud Platform ($324-583/month)
- **Enterprise**: AWS/Azure ($833-1,665/month)

## Usage

### Running Benchmarks

```bash
# Basic embedding benchmark
PYTHONPATH="components:bases" uv run python performance/benchmarks/benchmark_embeddings.py

# Full pipeline benchmark (recommended)
PYTHONPATH="components:bases" uv run python performance/benchmarks/benchmark_pipeline_simple.py

# Comprehensive analysis (advanced)
PYTHONPATH="components:bases" uv run python performance/benchmarks/benchmark_full_pipeline.py
```

### Environment Requirements

- Python 3.12+
- CUDA support (for GPU testing)
- Database connectivity (PostgreSQL + pgvector)
- Redis connectivity
- API access (Sejm API, ELI API)

## Benchmark Methodology

**Testing Approach:**

- End-to-end pipeline measurement
- Component-level performance isolation
- Real-world data processing scenarios
- Multiple run statistical analysis

**Test Environment:**

- Hardware: NVIDIA GTX 1060 6GB, x86_64 CPU
- Software: Python 3.12, PyTorch, HerBERT model
- Data: Polish legal documents and parliamentary proceedings

## Contributing

When adding new benchmarks:

1. Place scripts in `benchmarks/` directory
1. Save results in `results/` directory with timestamps
1. Update reports in `reports/` directory
1. Document methodology and findings
1. Include cost analysis for deployment decisions

## Related Documentation

- [Deployment Strategy](../deployments/k3s/README.md) - Kubernetes deployment options
- [Baremetal Deployment](../deployments/baremetal/README.md) - Bare metal installation
- [System Architecture](../IMPLEMENTATION_PLAN.md) - Overall system design
