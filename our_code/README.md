# Polymarket Trader Archetypes — Team Code

Unsupervised clustering of Polymarket wallets + KNN classification of archetype
membership from held-out features. All models implemented from scratch in NumPy;
DuckDB is used only for data aggregation.

## Layout

```
our_code/
  feature_extraction.py   # DuckDB -> per-wallet feature CSVs
  models.py               # KMeans, Hierarchical (Ward), KNN, metrics, PCA
  run_experiments.py      # Full pipeline runner
  build_notebook.py       # Generates notebooks/analysis.ipynb
  streamlit_app.py        # 3-page Streamlit demo
  README.md               # This file.
  notebooks/analysis.ipynb
  models/                 # (reserved for model pickles if we add any)
  streamlit/              # (reserved for app assets)
  results/                # Artifacts from run_experiments.py
```

## Data inputs

- `data/polymarket/trades/*.parquet` — CTF-exchange orderbook trades (~404M rows)
- `data/polymarket/legacy_trades/*.parquet` — FPMM AMM trades (~2M rows)
- `data/polymarket/markets/*.parquet` — market metadata (used for resolution prices)
- `data/polymarket/blocks/*.parquet` — block_number → timestamp lookup

Run the upstream `make setup` once to download and extract those (~36 GB
compressed, ~50 GB extracted).

## Usage

1. Build the feature matrix (this is the slow step: it scans ~404M rows):

   ```bash
   uv run python our_code/feature_extraction.py
   ```

   Output: `data/wallet_features_raw.csv` and
   `data/wallet_features_normalized.csv`. Wallets with < 10 trades are dropped.

2. Run the full experiment pipeline (K-Means sweep, Hierarchical, stability,
   KNN tuning, PCA):

   ```bash
   uv run python our_code/run_experiments.py
   ```

   Output: `our_code/results/*.csv`, `*.npy`, `summary.json`.

3. Regenerate the analysis notebook (optional — the repo ships with it):

   ```bash
   uv run python our_code/build_notebook.py
   uv run jupyter nbconvert --to notebook --execute our_code/notebooks/analysis.ipynb --inplace
   ```

4. Launch the Streamlit app:

   ```bash
   uv run streamlit run our_code/streamlit_app.py
   ```

## Feature split

**Clustering features** (used by K-Means / Hierarchical):
`trade_count, total_volume_usd, distinct_tokens, active_days, maker_fraction,
avg_trade_size, std_trade_size, trades_per_day, hour_entropy, buy_fraction,
avg_price, longshot_ratio`

**Classification features** (held out from clustering; used to predict the
archetype label via KNN):
`max_trade_size, trade_size_cv, win_rate, excess_return, total_pnl,
resolved_trade_count`

Skewed features are log-transformed before z-scoring; values are clipped to ±5σ.

## Disclaimer

This application does not provide trading advice.
