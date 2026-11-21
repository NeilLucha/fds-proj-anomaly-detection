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
