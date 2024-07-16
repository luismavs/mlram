# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: py312ba
#     language: python
#     name: python3
# ---

# %%
from sklearn.datasets import make_classification
import polars as pl


# %%
X, y = make_classification(
    n_samples=100_000, n_features=10, n_informative=5, n_classes=3
)


# %%
df = pl.from_numpy(X).with_columns(classification=y)
df.write_parquet("data/data.parquet")
