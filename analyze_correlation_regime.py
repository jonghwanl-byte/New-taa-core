"""
analyze_correlation_regime_v2.py

PRE-REGISTRATION #2 (follow-up to v1)
=============================================================================
Motivation (documented honestly — this is a follow-up test, not independent):
    v1 (analyze_correlation_regime.py) found QQQ-TLT correlation shifted from
    -0.198 (2015-2022) to +0.095 (2023-present), ACCEPTed under the v1
    criteria. But Period A in v1 lumps 2022 (a known outlier year where
    QQQ and TLT fell together on the inflation shock) together with the
    "normal" 2015-2021 regime. This test isolates 2022 as its own period to
    check whether the v1 result is driven by a genuine 2023+ regime shift,
    or by 2022 alone pulling the pre-split average down.

    IMPORTANT: this is a NEW pre-registration. The pass criteria below were
    fixed using the SAME thresholds as v1 (DELTA_THRESHOLD, ALPHA, BLOCK_SIZE)
    — not re-tuned based on v1's results — precisely to avoid post-hoc
    parameter selection.

Hypothesis:
    Split history into three regimes:
        A: 2015-01-01 -> 2021-12-31  ("pre-shock baseline")
        B: 2022-01-01 -> 2022-12-31  ("2022 inflation-shock outlier")
        C: 2023-01-01 -> today        ("post-2023 regime")
    For each asset pair, test all three pairwise deltas: A-B, B-C, A-C.

Test:
    Same moving-block bootstrap as v1 (block_size=BLOCK_SIZE, n_boot=N_BOOT).

Pass criteria (unchanged from v1, applied per period-comparison):
    1. |delta| >= DELTA_THRESHOLD
    2. Bootstrap CI (Bonferroni-adjusted alpha = ALPHA / N_COMPARISONS,
       since 3 period-pairs x 3 asset-pairs = 9 simultaneous comparisons)
       excludes 0.

Interpretation guide (fixed in advance):
    - If A-C is ACCEPT but A-B is NOT (i.e. 2022 alone doesn't explain the
      full A-vs-C gap) -> supports "2023+ is a genuine new regime", not just
      "2022 was an outlier that dragged the old average down."
    - If A-B ACCEPT and B-C ACCEPT with B roughly the midpoint of A and C ->
      consistent with 2022 being a transition year rather than 2023 being a
      distinct breakpoint.
    - If A-B ACCEPT but A-C is NOT -> the "regime change" in v1 was actually
      just the 2022 outlier, and 2023+ has reverted toward the old baseline.
      This would REVERSE the v1 conclusion and should be logged as such.
=============================================================================
"""

import json
import warnings
from datetime import datetime
from itertools import combinations
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
END = None

PERIODS = [
    ("A_2015_2021", "2015-01-01", "2022-01-01"),
    ("B_2022",       "2022-01-01", "2023-01-01"),
    ("C_2023_now",   "2023-01-01", None),
]

ROLLING_WINDOWS = [20, 60, 120, 200]   # diagnostic only, same as v1

BLOCK_SIZE = 20
N_BOOT = 5000
ALPHA = 0.05                  # will be Bonferroni-adjusted by N_COMPARISONS below
DELTA_THRESHOLD = 0.15        # unchanged from v1 — not re-tuned post-hoc

OUTPUT_DIR = Path("results")
RANDOM_SEED = 42

N_COMPARISONS = len(list(combinations(PERIODS, 2))) * len(PAIRS)  # 3 period-pairs x 3 asset-pairs = 9
ALPHA_ADJUSTED = ALPHA / N_COMPARISONS


# --------------------------------------------------------------------------
# DATA (identical to v1)
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


def slice_period(returns: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    if end is None:
        return returns[returns.index >= start]
    return returns[(returns.index >= start) & (returns.index < end)]


# --------------------------------------------------------------------------
# BLOCK BOOTSTRAP (identical logic to v1)
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
        corrs[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    return corrs


def compare_two_periods(returns_x: pd.DataFrame, label_1: str, ret_1: pd.DataFrame,
                         label_2: str, ret_2: pd.DataFrame, block_size: int,
                         n_boot: int, alpha_adj: float, delta_threshold: float,
                         rng: np.random.Generator) -> list[dict]:
    rows = []
    for a, b in PAIRS:
        x1, y1 = ret_1[a].values, ret_1[b].values
        x2, y2 = ret_2[a].values, ret_2[b].values

        r1_dist = block_bootstrap_corr(x1, y1, block_size, n_boot, rng)
        r2_dist = block_bootstrap_corr(x2, y2, block_size, n_boot, rng)
        delta_dist = r2_dist - r1_dist

        r1_point = np.corrcoef(x1, y1)[0, 1]
        r2_point = np.corrcoef(x2, y2)[0, 1]
        delta_point = r2_point - r1_point

        ci_lo, ci_hi = np.quantile(delta_dist, [alpha_adj / 2, 1 - alpha_adj / 2])
        ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        meets_threshold = abs(delta_point) >= delta_threshold
        verdict = "ACCEPT: differs" if (ci_excludes_zero and meets_threshold) else "REJECT: no confirmed diff"

        rows.append({
            "pair": f"{a}-{b}",
            "comparison": f"{label_1} -> {label_2}",
            "n_1": len(x1),
            "n_2": len(x2),
            "r_1": round(r1_point, 4),
            "r_2": round(r2_point, 4),
            "delta": round(delta_point, 4),
            "ci_low_adj": round(ci_lo, 4),
            "ci_high_adj": round(ci_hi, 4),
            "ci_excludes_zero": ci_excludes_zero,
            "meets_delta_threshold": meets_threshold,
            "verdict": verdict,
        })
    return rows


# --------------------------------------------------------------------------
# CHARTS (same style as v1, with two period-boundary lines)
# --------------------------------------------------------------------------
def plot_rolling_correlations(rolling: dict, boundaries: list, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for pair_key, df in rolling.items():
        fig, ax = plt.subplots(figsize=(11, 5))
        for col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=1.2)
        for label, dt in boundaries:
            if dt is not None:
                ax.axvline(pd.Timestamp(dt), color="black", linestyle="--", linewidth=1)
                ax.text(pd.Timestamp(dt), ax.get_ylim()[1], label, rotation=90,
                        fontsize=7, va="top")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_title(f"Rolling correlation with regime boundaries: {pair_key}")
        ax.set_ylabel("Pearson correlation")
        ax.legend(loc="upper left", fontsize=8, ncol=5)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"rolling_corr_v2_{pair_key}.png", dpi=140)
        plt.close(fig)


def rolling_correlations(returns: pd.DataFrame, windows) -> dict:
    out = {}
    for a, b in PAIRS:
        pair_key = f"{a}-{b}"
        out[pair_key] = pd.DataFrame(
            {f"w{w}": returns[a].rolling(w).corr(returns[b]) for w in windows}
        )
    return out


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
    print(f"Bonferroni-adjusted alpha: {ALPHA} / {N_COMPARISONS} comparisons = {ALPHA_ADJUSTED:.5f}")

    period_data = {label: slice_period(returns, start, end) for label, start, end in PERIODS}
    for label, df in period_data.items():
        print(f"  {label}: n={len(df)} ({df.index[0].date()} -> {df.index[-1].date()})")

    # Diagnostic charts
    rolling = rolling_correlations(returns, ROLLING_WINDOWS)
    boundaries = [(label, start) for label, start, _ in PERIODS]
    plot_rolling_correlations(rolling, boundaries, OUTPUT_DIR)

    # Confirmatory: all pairwise period comparisons, Bonferroni-adjusted
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    for (label_1, _, _), (label_2, _, _) in combinations(PERIODS, 2):
        rows = compare_two_periods(
            returns, label_1, period_data[label_1], label_2, period_data[label_2],
            BLOCK_SIZE, N_BOOT, ALPHA_ADJUSTED, DELTA_THRESHOLD, rng
        )
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUTPUT_DIR / "correlation_summary_v2_three_regime.csv", index=False)

    run_meta = {
        "run_at_utc": datetime.utcnow().isoformat(),
        "tickers": TICKERS,
        "periods": PERIODS,
        "block_size": BLOCK_SIZE,
        "n_boot": N_BOOT,
        "alpha_raw": ALPHA,
        "n_comparisons": N_COMPARISONS,
        "alpha_bonferroni_adjusted": ALPHA_ADJUSTED,
        "delta_threshold": DELTA_THRESHOLD,
        "seed": RANDOM_SEED,
        "note": "Follow-up pre-registration to v1; thresholds unchanged from v1 to avoid post-hoc tuning.",
    }
    with open(OUTPUT_DIR / "run_meta_v2.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print("\n=== Pre-registered 3-regime bootstrap test result (Bonferroni-adjusted) ===")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
