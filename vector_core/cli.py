"""Command Line Interface (CLI) for VectorCore."""

import argparse
import sys
import time
import numpy as np
import uvicorn

from vector_core.index.hnsw import HNSWIndex
from vector_core.index.flat import FlatIndex


def run_server(args: argparse.Namespace) -> None:
    """Launch the FastAPI REST server."""
    print(f"Starting VectorCore REST API on {args.host}:{args.port}...")
    uvicorn.run(
        "vector_core.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute an in-memory indexing and query latency benchmark."""
    print(f"\n--- Running VectorCore Benchmark ---")
    print(f"Vectors: {args.num_vectors:,} | Dimension: {args.dim} | Metric: {args.metric} | k: {args.k}\n")

    np.random.seed(42)
    vectors = np.random.randn(args.num_vectors, args.dim).astype(np.float32)
    ids = list(range(args.num_vectors))
    queries = np.random.randn(args.num_queries, args.dim).astype(np.float32)

    # 1. HNSW Index Benchmark
    print("Building HNSW index...")
    t0 = time.perf_counter()
    hnsw = HNSWIndex(dim=args.dim, metric=args.metric, m=16, ef_construction=64, ef_search=32)
    hnsw.add(ids, vectors)
    hnsw_build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for q in queries:
        hnsw.search(q, k=args.k)
    hnsw_query_time = time.perf_counter() - t0
    hnsw_qps = args.num_queries / hnsw_query_time if hnsw_query_time > 0 else 0

    # 2. Flat Index Benchmark (Ground Truth baseline)
    print("Building Flat baseline index...")
    t0 = time.perf_counter()
    flat = FlatIndex(dim=args.dim, metric=args.metric)
    flat.add(ids, vectors)
    flat_build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for q in queries:
        flat.search(q, k=args.k)
    flat_query_time = time.perf_counter() - t0
    flat_qps = args.num_queries / flat_query_time if flat_query_time > 0 else 0

    print("\nBenchmark Results:")
    print(f"{'Metric':<25} {'HNSW':<18} {'Flat (Exhaustive)':<18}")
    print("-" * 65)
    print(f"{'Build Time (s)':<25} {hnsw_build_time:<18.4f} {flat_build_time:<18.4f}")
    print(f"{'Query Time (s)':<25} {hnsw_query_time:<18.4f} {flat_query_time:<18.4f}")
    print(f"{'Throughput (QPS)':<25} {hnsw_qps:<18.1f} {flat_qps:<18.1f}")
    print("-" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vectorcore",
        description="VectorCore: High-performance HNSW vector search engine & REST service.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI vector database server.")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8001, help="Port number (default: 8001)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable hot-reload for development")

    # Subcommand: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run indexing and search benchmarks.")
    bench_parser.add_argument("--num-vectors", type=int, default=1000, help="Number of vectors (default: 1000)")
    bench_parser.add_argument("--num-queries", type=int, default=100, help="Number of query evaluations (default: 100)")
    bench_parser.add_argument("--dim", type=int, default=128, help="Vector dimension (default: 128)")
    bench_parser.add_argument("--k", type=int, default=10, help="Nearest neighbors k (default: 10)")
    bench_parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        choices=["l2", "cosine", "inner_product"],
        help="Distance metric (default: l2)",
    )

    args = parser.parse_args()

    if args.command == "serve":
        run_server(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()