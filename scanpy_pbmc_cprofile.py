"""Run the PBMC Scanpy workflow and profile rank_genes_groups with cProfile."""

import argparse
import cProfile
import pstats
from pathlib import Path

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

adata = sc.read_10x_mtx(
    data_dir / dataset / "filtered_gene_bc_matrices",
    var_names="gene_symbols",
    cache=True,
)
adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata)
sc.tl.pca(adata, svd_solver="arpack", n_comps=30)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.louvain(adata, resolution=0.5)
sc.tl.umap(adata, n_components=30)
adata.write(out_dir / f"{dataset}.scanpy.h5ad")

profile_path = out_dir / f"{dataset}_rank_genes.prof"
profile_text_path = out_dir / f"{dataset}_rank_genes_profile.txt"
profiler = cProfile.Profile()
profiler.enable()
sc.tl.rank_genes_groups(adata, "louvain", method="wilcoxon", use_raw=True)
profiler.disable()
profiler.dump_stats(profile_path)

with profile_text_path.open("w") as stream:
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

print(f"wrote cProfile data to {profile_path}")
print(f"wrote cProfile summary to {profile_text_path}")
