# FindWithin Benchmark Results

This file contains the performance metrics for document ingestion and search query latency.

## Run Benchmarks

To execute the benchmarks and record real values, ensure your Docker services are running (`docker-compose up -d`), then run the benchmark scripts:

```bash
# Run ingestion benchmarks
docker-compose exec backend python benchmarks/benchmark_ingestion.py

# Run search latency benchmarks
docker-compose exec backend python benchmarks/benchmark_search_latency.py
```

## Ingestion Benchmark

- **PDF Filename**: `benchmark_20p.pdf`
- **Pages**: 20
- **Chunks generated**: 20
- **Full ingestion pipeline processed 20 pages in 0.1995 seconds.**
- **Pages/sec**: 100.26
- **Chunks/sec**: 100.26
- **Embeddings/sec**: 100.26

## Search Latency Benchmark

| Chunk Count | Average Latency | p95 Latency |
|---|---:|---:|
| 100 chunks | 20.78 ms | 35.43 ms |
| 500 chunks | 19.73 ms | 22.96 ms |
| 1000 chunks | 20.55 ms | 22.19 ms |

## Retrieval Evaluation

- **Test queries**: 20
- **Correct result found in top 3**: 20
- **Recall@3**: 100.0%
