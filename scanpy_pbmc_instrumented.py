"""Run the PBMC Scanpy workflow and record elapsed time for each major section."""

import argparse
import time
from pathlib import Path

import pandas as pd
import scanpy as sc


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", default="data")
parser.add_argument("--data-set", default="pbmc3k")
parser.add_argument("--out-dir", default="results")
parser.add_argument("--num-threads", type=int, default=1)
args = parser.parse_args()

data_dir = Path(args.data_dir)
dataset = args.data_set
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")
sc.settings.n_jobs = args.num_threads
print(f"using {sc.settings.n_jobs} threads")

timings = []

start = time.perf_counter()
adata = sc.read_10x_mtx(
    data_dir / dataset / "filtered_gene_bc_matrices",
    var_names="gene_symbols",
    cache=True,
)
adata.var_names_make_unique()
timings.append({"dataset": dataset, "section": "read_data", "seconds": time.perf_counter() - start})

start = time.perf_counter()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
timings.append(
    {"dataset": dataset, "section": "filter_and_normalize", "seconds": time.perf_counter() - start}
)

start = time.perf_counter()
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
timings.append(
    {"dataset": dataset, "section": "highly_variable_genes", "seconds": time.perf_counter() - start}
)

start = time.perf_counter()
sc.pp.scale(adata)
timings.append({"dataset": dataset, "section": "scale", "seconds": time.perf_counter() - start})

start = time.perf_counter()
sc.tl.pca(adata, svd_solver="arpack", n_comps=30)
timings.append({"dataset": dataset, "section": "pca", "seconds": time.perf_counter() - start})

start = time.perf_counter()
sc.pp.neighbors(adata, n_pcs=30)
timings.append({"dataset": dataset, "section": "neighbors", "seconds": time.perf_counter() - start})

start = time.perf_counter()
sc.tl.louvain(adata, resolution=0.5)
timings.append({"dataset": dataset, "section": "louvain", "seconds": time.perf_counter() - start})

start = time.perf_counter()
sc.tl.umap(adata, n_components=30)
timings.append({"dataset": dataset, "section": "umap", "seconds": time.perf_counter() - start})

start = time.perf_counter()
adata.write(out_dir / f"{dataset}.scanpy.h5ad")
timings.append({"dataset": dataset, "section": "write_h5ad", "seconds": time.perf_counter() - start})

start = time.perf_counter()
sc.tl.rank_genes_groups(adata, "louvain", method="wilcoxon", use_raw=True)
timings.append(
    {"dataset": dataset, "section": "rank_genes_groups", "seconds": time.perf_counter() - start}
)

timing_path = out_dir / f"{dataset}_section_times.csv"
pd.DataFrame(timings).to_csv(timing_path, index=False)
print(f"wrote section timings to {timing_path}")
