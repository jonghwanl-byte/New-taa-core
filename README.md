# QQQ / TLT / GLD Correlation Regime Test

Pre-registered test of whether the pairwise return correlation of
QQQ, TLT, and GLD has structurally changed since **2023-01-01**,
motivated by a possible post-2022 macro regime shift (fiscal
dominance / government-debt-driven liquidity, described e.g. in the
2026-09-11 Infomax Live segment with 성상현).

## Why this exists

Core TAA allocation (60/20/20 QQQ/TLT/GLD) does not explicitly use
correlation — allocation is driven by independent per-asset MA
scores. But if the historical diversification benefit between these
three assets has broken down, that's worth knowing even if it
doesn't automatically imply a rule change. This repo answers the
narrow empirical question only: **did the correlation actually
change, and is the change statistically supported?**

Nothing here modifies `taa/engine.py` or any live signal bot. This
is a standalone research check with a hard pre-registration gate —
see the docstring in `analyze_correlation_regime.py` for the exact
hypothesis, test, and accept/reject thresholds fixed *before* running.

## Method

1. **Diagnostic (not decision-relevant):** rolling Pearson
   correlation for each pair over 20/60/120/200-day windows, plotted
   with the 2023-01-01 split marked. Multiple windows are used
   together because a single short window (e.g. 20d) is noise-prone
   (SE ≈ 1/√N ⇒ ~0.22 at N=20 vs ~0.09 at N=120).
2. **Confirmatory (decision-relevant):** moving-block bootstrap
   (block = 20 trading days, preserves short-horizon autocorrelation)
   comparing full-period correlation pre- vs post-2023-01-01.
   Accept "regime changed" for a pair only if **both**:
   - `|Δr| ≥ 0.15`, and
   - the 95% bootstrap CI for `Δr` excludes 0.

Both criteria were fixed before any results were viewed. If a pair
fails either one, it is logged as **rejected** — not reinterpreted.

## Usage

```bash
pip install -r requirements.txt
python analyze_correlation_regime.py
```

Outputs (written to `results/`):
- `correlation_summary.csv` — the pre-registered test result per pair
- `rolling_corr_<PAIR>.csv` / `.png` — diagnostic rolling correlations
- `run_meta.json` — exact parameters used for this run (reproducibility)

## Running via GitHub Actions

`.github/workflows/correlation_analysis.yml` runs the script on
manual dispatch and uploads `results/` as a workflow artifact — no
schedule, no Telegram delivery, since this is a one-off/occasional
research check rather than a live signal.

## Interpreting the result

- **ACCEPT** on a pair: the correlation shift is both economically
  meaningful and statistically supported by this test. This is a
  trigger to *investigate further* (e.g. does it change any backtest
  conclusion under `taa/engine.py`), not an automatic rule change —
  the existing bar for rule changes still applies.
- **REJECT**: log it. Do not re-run with different `SPLIT_DATE`,
  `BLOCK_SIZE`, or `DELTA_THRESHOLD` values to chase significance —
  that defeats the point of pre-registration. If you want to test a
  different split date or threshold, that's a new pre-registered
  hypothesis, documented as such.
