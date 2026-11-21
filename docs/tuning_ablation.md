# Hyperparameter Tuning and Ablation Guide

## 1. Key Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `window_size` | $L$ | 200 | Size of rolling reference window |
| `candidate_size` | $M$ | 50 | Number of observations in post-drift candidate window |
| `tau` | $\tau$ | 3.0 | Anomaly alert multiplier ($S_t > \tau$) |
| `tau_cand` | $\tau_{\text{cand}}$ | 3.0 | Point threshold for candidate window contamination test |
| `rho_max` | $\rho_{\text{max}}$ | 0.30 | Max allowable fraction of contaminated points in candidate |
| `kappa_max` | $\kappa_{\text{max}}$ | 3.0 | Max allowable ratio of candidate MAD to baseline MAD |
| `freeze_window` | $L_0$ | 200 | Initial window size for Version C |
| `hold_steps` | $H$ | 50 | Hold buffer length in Version C delayed adaptation |
| `adwin_delta` | $\delta$ | 0.002 | ADWIN confidence / false alarm probability |

---

## 2. Ablation Analysis Strategy

### Versions E and E2

E adds a robust MAD/IQR/absolute scale floor, a causal trend predictor, an optional warm-up-selected seasonal lag, causal score-quantile threshold scaling, 2-of-K persistence, and a dispersion-gated drift reset. Its bounded validation ablation is available in the real benchmark's `--series-split` mode; it searches scale-floor ratio, fixed versus automatic lag, persistence, and low/high thresholds. The report records pointwise F1, positive-series recall, zero-event macro FPR, and selected validation error cases.

E2 retains E's causal score and adds dual-tier alerting: an extreme shock alerts immediately, while a moderate score needs at least two threshold crossings in K=2 or 3 steps. Its validation grid searches scale-floor ratio, lag mode, K, and moderate/extreme threshold pair. Selection maximizes `0.5 * pooled point F1 + 0.5 * positive-series macro Event F1`, subject to zero-event macro FPR `<= 0.10`. If no candidate passes the cap, output explicitly reports a lowest-FPR fallback. E2's validation score report summarizes one-point spikes, wide-event interiors, and false-alert clusters on zero-event series. Event labels only select which validation scores to summarize; they do not affect online scores. Both variants use a capped subset of at most ten positive and five zero-event series for ablation, so tuning is quick but not exhaustive.

Default-threshold validation diagnostics on the current local data inspected 30 NAB and 34 Yahoo series. Yahoo one-point spike score median was 0.40 and its 90th percentile 8.08 (14.3% above the default extreme threshold 7.0); 367 wide-event interior points had 90th percentile 0 and 99th percentile 3.52. NAB's sample had 107 zero-event false-alert cluster peaks with median 4.24 and 90th percentile 480. These are pre-tuning diagnostic samples, not benchmark performance results; they show that one shared extreme threshold is unlikely to separate Yahoo spikes from all NAB control bursts.

Run the comparison with:

```bash
python scripts/run_real_benchmarks.py --dataset nab --series-split --output-dir outputs/final_comparison
```

This selects E and E2 settings on validation-side series and scores on held-out series. The bounded search is exploratory; repeat with another split before treating a configuration as general.

The benchmark shows progress bars for threshold grids, E candidate combinations, the per-series evaluation loop, and LSTM training epochs. A–D use the fixed threshold grid `[1.5, 2.0, ..., 5.0]`; their remaining parameters are config values, not searched. LSTM uses fixed lookback, hidden width, epoch count, and learning rate. Its threshold is calibrated from validation residuals. `drift-anomaly evaluate` does no tuning; `drift-anomaly tune` searches only `tau` for one series.

1. **Version A vs. Version B (Contamination Gate Ablation)**:
   - Version B continuously rolls over incoming data regardless of contamination.
   - When an anomaly burst occurs, Version B absorbs outliers into its rolling window, inflating $\text{MAD}_t$ and suppressing subsequent anomaly scores.
   - Version A detects the shift, checks the candidate window, detects high contamination ($\rho > \rho_{\text{max}}$ or $\kappa > \kappa_{\text{max}}$), and rejects adaptation, preventing baseline corruption.

2. **Version A vs. Version C (Rolling vs. Frozen Baseline)**:
   - Version C keeps a static baseline until a verified drift occurs.
   - Ideal for piecewise-stationary signals with sharp, well-separated changepoints.
   - Version A adapts flexibly to gradual transitions while remaining protected against bursts.

3. **ADWIN vs. Page-Hinkley**:
   - ADWIN adapts subwindow sizes dynamically with theoretical confidence bounds.
   - Page-Hinkley offers lightweight sequential CUSUM detection with minimal computational overhead.
