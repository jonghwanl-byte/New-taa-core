"""
analyze_correlation_regime.py

Pre-registered test: has the pairwise correlation structure of
QQQ / TLT / GLD changed since 2023-01-01 relative to the prior period?

=============================================================================
PRE-REGISTRATION (fixed BEFORE results are viewed — do not edit after running)
=============================================================================
Hypothesis:
    The full-period Pearson correlation of daily returns for each pair
    (QQQ-TLT, QQQ-GLD, TLT-GLD) computed over PERIOD_B (2023-01-01 -> today)
    differs from the correlation computed over PERIOD_A (START -> 2022-12-31).

Test:
    Moving-block bootstrap (block size = BLOCK_SIZE trading days, to preserve
    short-term autocorrelation in return series) with N_BOOT resamples per
    period. For each pair, compute r_A (bootstrap dist) and r_B (bootstrap
    dist) independently, then the distribution of delta = r_B - r_A.

Pass criteria (ACCEPT "regime changed" for a pair only if BOTH hold):
    1. |mean(delta)| >= DELTA_THRESHOLD   (economically meaningful shift)
    2. The (1-ALPHA) bootstrap CI for delta excludes 0 (statistically supported)

If a pair fails either criterion -> REJECT for that pair, log it, and do NOT
act on it (no rule changes) without a documented reinvestigation trigger.

Rolling correlation charts (multiple windows) are DIAGNOSTIC ONLY — for
visual regime inspection — and are not themselves part of the accept/reject
decision, per the discussion that a single-window correlation reading is
noise-prone.
=============================================================================
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
TICKERS = ["QQQ", "TLT", "GLD"]
PAIRS = [("QQQ", "TLT"), ("QQQ", "GLD"), ("TLT", "GLD")]

START = "2015-01-01"
SPLIT_DATE = "2023-01-01"   # PERIOD_A = [START, SPLIT_DATE) / PERIOD_B = [SPLIT_DATE, today]
END = None                  # None = through latest available

ROLLING_WINDOWS = [20, 60, 120, 200]   # diagnostic only

BLOCK_SIZE = 20              # trading days per bootstrap block (~1 month)
N_BOOT = 5000
ALPHA = 0.05
DELTA_THRESHOLD = 0.15       # minimum |change in correlation| to call it "changed"

OUTPUT_DIR = Path("results")
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def fetch_prices(tickers, start, end=None):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"]
    else:
        px = data[["Close"]]
        px.columns = tickers
    px = px.dropna(how="all").ffill().dropna()
    return px


def compute_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()


# --------------------------------------------------------------------------
# ROLLING CORRELATION (diagnostic)
# --------------------------------------------------------------------------
def rolling_correlations(returns: pd.DataFrame, windows) -> dict:
    out = {}
    for a, b in PAIRS:
        pair_key = f"{a}-{b}"
        out[pair_key] = pd.DataFrame(
            {f"w{w}": returns[a].rolling(w).corr(returns[b]) for w in windows}
        )
    return out


# --------------------------------------------------------------------------
# BLOCK BOOTSTRAP TEST (confirmatory)
# --------------------------------------------------------------------------
def block_bootstrap_corr(x: np.ndarray, y: np.ndarray, block_size: int,
                          n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    if max_start <= 0:
        raise ValueError("Series shorter than one block; reduce BLOCK_SIZE.")

    corrs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        xs, ys = x[idx], y[idx]
        corrs[i] = np.corrcoef(xs, ys)[0, 1]
    return corrs


def run_bootstrap_test(returns: pd.DataFrame, split_date: str,
                        block_size: int, n_boot: int, alpha: float,
                        delta_threshold: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    period_a = returns[returns.index < split_date]
    period_b = returns[returns.index >= split_date]

    rows = []
    for a, b in PAIRS:
        xa, ya = period_a[a].values, period_a[b].values
        xb, yb = period_b[a].values, period_b[b].values

        r_a_dist = block_bootstrap_corr(xa, ya, block_size, n_boot, rng)
        r_b_dist = block_bootstrap_corr(xb, yb, block_size, n_boot, rng)
        delta_dist = r_b_dist - r_a_dist

        r_a_point = np.corrcoef(xa, ya)[0, 1]
        r_b_point = np.corrcoef(xb, yb)[0, 1]
        delta_point = r_b_point - r_a_point

        ci_lo, ci_hi = np.quantile(delta_dist, [alpha / 2, 1 - alpha / 2])
        ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        meets_threshold = abs(delta_point) >= delta_threshold
        verdict = "ACCEPT: regime changed" if (ci_excludes_zero and meets_threshold) else "REJECT: no confirmed change"

        rows.append({
            "pair": f"{a}-{b}",
            "n_period_a": len(xa),
            "n_period_b": len(xb),
            "r_period_a": round(r_a_point, 4),
            "r_period_b": round(r_b_point, 4),
            "delta": round(delta_point, 4),
            "ci_low_95": round(ci_lo, 4),
            "ci_high_95": round(ci_hi, 4),
            "ci_excludes_zero": ci_excludes_zero,
            "meets_delta_threshold": meets_threshold,
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# CHARTS
# --------------------------------------------------------------------------
def plot_rolling_correlations(rolling: dict, split_date: str, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for pair_key, df in rolling.items():
        fig, ax = plt.subplots(figsize=(11, 5))
        for col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=1.2)
        ax.axvline(pd.Timestamp(split_date), color="black", linestyle="--", linewidth=1, label=split_date)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_title(f"Rolling correlation: {pair_key}")
        ax.set_ylabel("Pearson correlation")
        ax.legend(loc="upper left", fontsize=8, ncol=5)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"rolling_corr_{pair_key}.png", dpi=140)
        plt.close(fig)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Fetching {TICKERS} from {START} to {END or 'latest'}...")
    px = fetch_prices(TICKERS, START, END)
    returns = compute_returns(px)
    print(f"Loaded {len(returns)} daily return observations "
          f"({returns.index[0].date()} -> {returns.index[-1].date()})")

    # Diagnostic: rolling correlations across multiple windows
    rolling = rolling_correlations(returns, ROLLING_WINDOWS)
    for pair_key, df in rolling.items():
        df.to_csv(OUTPUT_DIR / f"rolling_corr_{pair_key}.csv")
    plot_rolling_correlations(rolling, SPLIT_DATE, OUTPUT_DIR)

    # Confirmatory: pre-registered block bootstrap test
    summary = run_bootstrap_test(
        returns, SPLIT_DATE, BLOCK_SIZE, N_BOOT, ALPHA, DELTA_THRESHOLD, RANDOM_SEED
    )
    summary.to_csv(OUTPUT_DIR / "correlation_summary.csv", index=False)

    run_meta = {
        "run_at_utc": datetime.utcnow().isoformat(),
        "tickers": TICKERS,
        "start": START,
        "split_date": SPLIT_DATE,
        "end": returns.index[-1].strftime("%Y-%m-%d"),
        "rolling_windows": ROLLING_WINDOWS,
        "block_size": BLOCK_SIZE,
        "n_boot": N_BOOT,
        "alpha": ALPHA,
        "delta_threshold": DELTA_THRESHOLD,
        "seed": RANDOM_SEED,
    }
    with open(OUTPUT_DIR / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print("\n=== Pre-registered bootstrap test result ===")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
