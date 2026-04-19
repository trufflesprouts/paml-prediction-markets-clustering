"""Per-wallet feature extraction from Polymarket parquet data.

Two-pass strategy (disk-friendly):
    Pass 1: cheap count-per-wallet over UNION ALL of maker/taker/trader.
            HAVING COUNT(*) >= MIN_TRADES.  Project only `wallet` -> aggregation
            is slim.
    Python: uniformly sample SAMPLE_N wallets (seed=42) from the filtered set.
    Pass 2: main aggregation restricted to the sampled wallets via semi-joins
            pushed down into each UNION ALL leg.
    Pass 3: hour_entropy aggregation on the same sampled wallets.

Output:
    our_code/data/wallet_features_raw.csv
    our_code/data/wallet_features_normalized.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# `data/` holds the 33 GB raw parquet (gitignored); precomputed wallet features
# are written into `our_code/data/` so they can be committed.
DATA_DIR = ROOT / "data"
POLY = DATA_DIR / "polymarket"

OUT_DIR = ROOT / "our_code" / "data"
RAW_OUT = OUT_DIR / "wallet_features_raw.csv"
NORM_OUT = OUT_DIR / "wallet_features_normalized.csv"

MIN_TRADES = 10
SAMPLE_N = 100_000
SAMPLE_SEED = 42
CLIP_STD = 5.0

LOG_FEATURES = [
    "trade_count", "total_volume_usd", "distinct_tokens", "active_days",
    "avg_trade_size", "std_trade_size", "max_trade_size",
    "trades_per_day", "resolved_trade_count",
]

CLUSTER_FEATURES = [
    "trade_count", "total_volume_usd", "distinct_tokens", "active_days",
    "maker_fraction", "avg_trade_size", "std_trade_size", "trades_per_day",
    "hour_entropy", "buy_fraction", "avg_price", "longshot_ratio",
]

CLASSIFY_FEATURES = [
    "max_trade_size", "trade_size_cv", "win_rate",
    "excess_return", "total_pnl", "resolved_trade_count",
]


def log1p_signed(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------- SQL builders

def count_sql() -> str:
    trades_glob = str(POLY / "trades" / "*.parquet")
    legacy_glob = str(POLY / "legacy_trades" / "*.parquet")
    return f"""
    SELECT wallet, COUNT(*) AS trade_count
    FROM (
        SELECT maker AS wallet FROM '{trades_glob}'
        UNION ALL
        SELECT taker AS wallet FROM '{trades_glob}'
        UNION ALL
        SELECT trader AS wallet FROM '{legacy_glob}'
    )
    GROUP BY wallet
    HAVING COUNT(*) >= {MIN_TRADES}
    """


def setup_small_tables(con: duckdb.DuckDBPyConnection) -> None:
    blocks_glob = str(POLY / "blocks" / "*.parquet")
    markets_glob = str(POLY / "markets" / "*.parquet")

    print("  [setup] building blocks_ts lookup ...")
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE blocks_ts AS
        SELECT block_number, CAST(timestamp AS TIMESTAMP) AS ts
        FROM '{blocks_glob}';
    """)
    print(f"    done ({time.time()-t0:.1f}s)")

    print("  [setup] building token_resolution lookup ...")
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE token_resolution AS
        WITH parsed AS (
            SELECT
                condition_id,
                market_maker_address,
                CAST(outcome_prices AS JSON) AS prices_json,
                CAST(clob_token_ids AS JSON) AS tokens_json
            FROM '{markets_glob}'
            WHERE closed = TRUE
              AND clob_token_ids IS NOT NULL
              AND outcome_prices IS NOT NULL
        ),
        unnested AS (
            SELECT
                p.condition_id,
                p.market_maker_address,
                idx,
                CAST(json_extract_string(p.tokens_json, '$[' || idx || ']') AS VARCHAR) AS token_id,
                TRY_CAST(json_extract_string(p.prices_json, '$[' || idx || ']') AS DOUBLE) AS resolved_price
            FROM parsed p,
                 generate_series(0, CAST(json_array_length(p.tokens_json) AS INTEGER) - 1) AS g(idx)
        ),
        resolved AS (
            SELECT condition_id FROM unnested
            GROUP BY condition_id HAVING MAX(resolved_price) > 0.99
        )
        SELECT u.condition_id, u.market_maker_address, u.idx, u.token_id, u.resolved_price
        FROM unnested u JOIN resolved rm USING (condition_id)
        WHERE u.token_id IS NOT NULL;
    """)
    n = con.execute("SELECT COUNT(*) FROM token_resolution").fetchone()[0]
    print(f"    token_resolution rows: {n:,} ({time.time()-t0:.1f}s)")


def main_aggregation_sql_sampled() -> str:
    """Main agg restricted to wallets in TEMP table `sampled_wallets(wallet)`."""
    trades_glob = str(POLY / "trades" / "*.parquet")
    legacy_glob = str(POLY / "legacy_trades" / "*.parquet")
    return f"""
    WITH per_trade AS (
        SELECT
            t.maker AS wallet,
            1 AS is_maker,
            t.block_number,
            CASE
                WHEN t.maker_asset_id = '0' AND t.taker_amount > 0
                    THEN CAST(t.maker_amount AS DOUBLE) / 1e6
                WHEN t.taker_asset_id = '0' AND t.maker_amount > 0
                    THEN CAST(t.taker_amount AS DOUBLE) / 1e6
            END AS size_usd,
            CASE
                WHEN t.maker_asset_id = '0' AND t.taker_amount > 0
                    THEN CAST(t.maker_amount AS DOUBLE) / CAST(t.taker_amount AS DOUBLE)
                WHEN t.taker_asset_id = '0' AND t.maker_amount > 0
                    THEN CAST(t.taker_amount AS DOUBLE) / CAST(t.maker_amount AS DOUBLE)
            END AS price,
            CASE
                WHEN t.maker_asset_id = '0' THEN 1
                WHEN t.taker_asset_id = '0' THEN 0
            END AS is_buy,
            CASE
                WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                WHEN t.taker_asset_id = '0' THEN t.maker_asset_id
            END AS token_id,
            CAST(NULL AS VARCHAR) AS fpmm_address,
            CAST(NULL AS BIGINT) AS outcome_index
        FROM '{trades_glob}' t
        SEMI JOIN sampled_wallets s ON s.wallet = t.maker

        UNION ALL

        SELECT
            t.taker AS wallet,
            0 AS is_maker,
            t.block_number,
            CASE
                WHEN t.taker_asset_id = '0' AND t.maker_amount > 0
                    THEN CAST(t.taker_amount AS DOUBLE) / 1e6
                WHEN t.maker_asset_id = '0' AND t.taker_amount > 0
                    THEN CAST(t.maker_amount AS DOUBLE) / 1e6
            END AS size_usd,
            CASE
                WHEN t.taker_asset_id = '0' AND t.maker_amount > 0
                    THEN CAST(t.taker_amount AS DOUBLE) / CAST(t.maker_amount AS DOUBLE)
                WHEN t.maker_asset_id = '0' AND t.taker_amount > 0
                    THEN CAST(t.maker_amount AS DOUBLE) / CAST(t.taker_amount AS DOUBLE)
            END AS price,
            CASE
                WHEN t.taker_asset_id = '0' THEN 1
                WHEN t.maker_asset_id = '0' THEN 0
            END AS is_buy,
            CASE
                WHEN t.taker_asset_id = '0' THEN t.maker_asset_id
                WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
            END AS token_id,
            CAST(NULL AS VARCHAR) AS fpmm_address,
            CAST(NULL AS BIGINT) AS outcome_index
        FROM '{trades_glob}' t
        SEMI JOIN sampled_wallets s ON s.wallet = t.taker

        UNION ALL

        SELECT
            l.trader AS wallet,
            0 AS is_maker,
            l.block_number,
            -- Legacy FPMM collateral uses either USDC (6 decimals) or 18-decimal
            -- tokens depending on the pool.  Detect per-trade by magnitude.
            CASE
                WHEN TRY_CAST(l.amount AS DOUBLE) > 1e15
                    THEN TRY_CAST(l.amount AS DOUBLE) / 1e18
                ELSE TRY_CAST(l.amount AS DOUBLE) / 1e6
            END AS size_usd,
            -- price = amount / outcome_tokens.  Both values scale together in
            -- each trade, so the ratio survives either decimal convention.
            CASE
                WHEN TRY_CAST(l.outcome_tokens AS DOUBLE) > 0
                    THEN TRY_CAST(l.amount AS DOUBLE) / TRY_CAST(l.outcome_tokens AS DOUBLE)
            END AS price,
            CASE WHEN l.is_buy THEN 1 ELSE 0 END AS is_buy,
            CAST(NULL AS VARCHAR) AS token_id,
            l.fpmm_address,
            l.outcome_index
        FROM '{legacy_glob}' l
        SEMI JOIN sampled_wallets s ON s.wallet = l.trader
    ),
    enriched AS (
        SELECT
            pt.wallet, pt.is_maker, pt.is_buy, pt.size_usd, pt.price, pt.token_id,
            bt.ts,
            COALESCE(tr_ctf.resolved_price, tr_leg.resolved_price) AS resolved_price
        FROM per_trade pt
        LEFT JOIN blocks_ts bt USING (block_number)
        LEFT JOIN token_resolution tr_ctf ON tr_ctf.token_id = pt.token_id
        LEFT JOIN token_resolution tr_leg
              ON tr_leg.market_maker_address = pt.fpmm_address
             AND tr_leg.idx = pt.outcome_index
    )
    SELECT
        wallet,
        COUNT(*) AS trade_count,
        SUM(COALESCE(size_usd, 0)) AS total_volume_usd,
        COUNT(DISTINCT token_id) AS distinct_tokens,
        COUNT(DISTINCT CAST(ts AS DATE)) AS active_days,
        AVG(CAST(is_maker AS DOUBLE)) AS maker_fraction,
        AVG(size_usd) AS avg_trade_size,
        STDDEV_SAMP(size_usd) AS std_trade_size,
        MAX(size_usd) AS max_trade_size,
        AVG(CASE WHEN is_buy = 1 THEN 1.0 WHEN is_buy = 0 THEN 0.0 END) AS buy_fraction,
        AVG(price) AS avg_price,
        AVG(CASE WHEN is_buy = 1 AND price < 0.10 THEN 1.0
                 WHEN is_buy = 1 THEN 0.0 END) AS longshot_ratio,
        COUNT(CASE WHEN resolved_price IS NOT NULL THEN 1 END) AS resolved_trade_count,
        AVG(CASE
            WHEN resolved_price IS NOT NULL AND is_buy = 1 AND resolved_price > 0.99 THEN 1.0
            WHEN resolved_price IS NOT NULL AND is_buy = 1 THEN 0.0
        END) AS win_rate,
        AVG(CASE
            WHEN resolved_price IS NOT NULL AND is_buy = 1 AND price IS NOT NULL AND price > 0
            THEN (resolved_price - price) / price
        END) AS excess_return,
        SUM(CASE
            WHEN resolved_price IS NOT NULL AND is_buy = 1 AND price IS NOT NULL AND price > 0
            THEN size_usd * (resolved_price - price) / price
            ELSE 0
        END) AS total_pnl
    FROM enriched
    GROUP BY wallet
    """


def hour_entropy_sql_sampled() -> str:
    trades_glob = str(POLY / "trades" / "*.parquet")
    legacy_glob = str(POLY / "legacy_trades" / "*.parquet")
    return f"""
    WITH all_actors AS (
        SELECT t.maker AS wallet, t.block_number
        FROM '{trades_glob}' t
        SEMI JOIN sampled_wallets s ON s.wallet = t.maker
        UNION ALL
        SELECT t.taker AS wallet, t.block_number
        FROM '{trades_glob}' t
        SEMI JOIN sampled_wallets s ON s.wallet = t.taker
        UNION ALL
        SELECT l.trader AS wallet, l.block_number
        FROM '{legacy_glob}' l
        SEMI JOIN sampled_wallets s ON s.wallet = l.trader
    ),
    hourly AS (
        SELECT a.wallet,
               EXTRACT(HOUR FROM bt.ts) AS hr,
               COUNT(*)::DOUBLE AS c
        FROM all_actors a
        JOIN blocks_ts bt USING (block_number)
        WHERE bt.ts IS NOT NULL
        GROUP BY a.wallet, EXTRACT(HOUR FROM bt.ts)
    ),
    probs AS (
        SELECT wallet, hr, c / SUM(c) OVER (PARTITION BY wallet) AS p
        FROM hourly
    )
    SELECT wallet, SUM(-(p * LN(p))) / LN(24) AS hour_entropy
    FROM probs WHERE p > 0
    GROUP BY wallet
    """


# --------------------------------------------------------------- normalization

def normalize(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if col in LOG_FEATURES:
            out[col] = log1p_signed(out[col])
        mu = out[col].mean()
        sd = out[col].std(ddof=0)
        out[col] = (out[col] - mu) / sd if sd > 0 else 0.0
        out[col] = out[col].clip(-CLIP_STD, CLIP_STD)
    return out


# ---------------------------------------------------------------- main driver

def main() -> None:
    t0 = time.time()
    print(f"[feature_extraction] starting (data dir: {DATA_DIR})", flush=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='10GB';")
    con.execute("PRAGMA preserve_insertion_order=false;")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_tmp';")

    # -------- Pass 1: count trades per wallet, HAVING COUNT >= MIN_TRADES
    print(f"[pass 1] counting trades per wallet (min {MIN_TRADES}) ...", flush=True)
    t1 = time.time()
    wc_df = con.execute(count_sql()).fetchdf()
    print(f"  pass 1: {len(wc_df):,} wallets with >= {MIN_TRADES} trades in {time.time()-t1:.1f}s", flush=True)

    # -------- Python: sample SAMPLE_N wallets at seed=42
    rng = np.random.default_rng(SAMPLE_SEED)
    n_avail = len(wc_df)
    n_sample = min(SAMPLE_N, n_avail)
    idx = rng.choice(n_avail, size=n_sample, replace=False)
    sampled = wc_df.iloc[idx][["wallet"]].reset_index(drop=True)
    print(f"[sample] sampled {n_sample:,} wallets (seed={SAMPLE_SEED}) from {n_avail:,} candidates", flush=True)

    con.register("sampled_df", sampled)
    con.execute("CREATE OR REPLACE TEMP TABLE sampled_wallets AS SELECT wallet FROM sampled_df")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samp ON sampled_wallets(wallet)")

    # -------- Pass 2: small-table setup + main aggregation
    setup_small_tables(con)

    print(f"[pass 2] main aggregation (sampled) ...", flush=True)
    t1 = time.time()
    df = con.execute(main_aggregation_sql_sampled()).fetchdf()
    print(f"  pass 2: {len(df):,} wallets aggregated in {time.time()-t1:.1f}s", flush=True)

    # -------- Pass 3: hour_entropy
    print(f"[pass 3] hour_entropy ...", flush=True)
    t1 = time.time()
    hdf = con.execute(hour_entropy_sql_sampled()).fetchdf()
    print(f"  pass 3: {len(hdf):,} entropy rows in {time.time()-t1:.1f}s", flush=True)

    df = df.merge(hdf, on="wallet", how="left")
    df["hour_entropy"] = df["hour_entropy"].fillna(0.0)

    # fill NaNs
    for col, fill in [
        ("avg_trade_size", 0.0), ("std_trade_size", 0.0), ("max_trade_size", 0.0),
        ("buy_fraction", 0.5), ("avg_price", 0.5), ("longshot_ratio", 0.0),
        ("win_rate", 0.0), ("excess_return", 0.0), ("total_pnl", 0.0),
    ]:
        df[col] = df[col].fillna(fill)
    df["distinct_tokens"] = df["distinct_tokens"].fillna(0).astype(np.int64)

    df["trades_per_day"] = df["trade_count"] / df["active_days"].clip(lower=1)
    df["trade_size_cv"] = (df["std_trade_size"] / df["avg_trade_size"].replace(0, np.nan)).fillna(0.0)

    keep = [
        "wallet",
        "trade_count", "total_volume_usd", "distinct_tokens", "active_days",
        "maker_fraction", "avg_trade_size", "std_trade_size", "max_trade_size",
        "trades_per_day", "hour_entropy", "buy_fraction", "avg_price", "longshot_ratio",
        "resolved_trade_count", "win_rate", "excess_return", "total_pnl",
        "trade_size_cv",
    ]
    df = df[keep]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_OUT, index=False)
    print(f"[feature_extraction] wrote raw -> {RAW_OUT}  ({len(df):,} wallets)", flush=True)

    all_features = list(dict.fromkeys(CLUSTER_FEATURES + CLASSIFY_FEATURES))
    norm = df[["wallet"] + all_features].copy()
    norm_values = normalize(norm[all_features], all_features)
    norm = pd.concat([norm[["wallet"]].reset_index(drop=True),
                      norm_values.reset_index(drop=True)], axis=1)
    norm.to_csv(NORM_OUT, index=False)
    print(f"[feature_extraction] wrote normalized -> {NORM_OUT}", flush=True)

    print("\n[feature_extraction] Summary (raw):", flush=True)
    print(df[all_features].describe().T[["mean", "std", "min", "max"]].to_string())
    print(f"\n[feature_extraction] total elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
