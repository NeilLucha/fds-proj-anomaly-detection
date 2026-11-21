# Mathematical and Algorithmic Specification

## 1. Sequential Univariate Setting & Causality Guarantee
Let a univariate time series be $\{x_t\}_{t \ge 1}$, where $x_t \in \mathbb{R}$ arrives sequentially.
Online anomaly detection requires strict temporal causality:
- At time $t$, when computing scale estimates, median, drift conditions, or anomaly scores for $x_t$, **only** historical observations prior to $t$ ($\{x_1, \dots, x_{t-1}\}$) are accessible.
- Anomaly labels $y_t \in \{0, 1\}$ are never accessible during online scoring, drift detection, candidate window validation, or baseline adaptation. Labels are reserved strictly for offline evaluation.

---

## 2. Robust Location, Scale, and Anomaly Score

Given a causal reference window $W_t^{ref} = (x_{t-L}, \dots, x_{t-1})$ of length $L$:

### 2.1 Rolling Median
$$m_t = \text{median}(W_t^{ref})$$

### 2.2 Median Absolute Deviation (MAD)
$$\text{MAD}_t = \text{median}\left(\left\{ |x_i - m_t| : x_i \in W_t^{ref} \right\}\right)$$

### 2.3 Consistent Scale Estimate $\hat{\sigma}_t$
For standard Gaussian distributions, $\text{MAD} \to \Phi^{-1}(0.75) \cdot \sigma \approx 0.67449 \cdot \sigma$. To ensure asymptotic consistency with Gaussian standard deviation $\sigma$:
$$c = \frac{1}{\Phi^{-1}(0.75)} \approx 1.4826$$

With a numerical stability guard $\epsilon > 0$ (default $10^{-6}$):
$$\hat{\sigma}_t = c \cdot \text{MAD}_t + \epsilon$$

### 2.4 Anomaly Score & Alert
$$S_t = \frac{|x_t - m_t|}{\hat{\sigma}_t}$$

$$\hat{y}_t = \begin{cases} 1 & \text{if } S_t > \tau \\ 0 & \text{otherwise} \end{cases}$$
where $\tau > 0$ is tuned strictly on validation splits.

---

## 3. Drift Detection Algorithms

### 3.1 ADWIN (Adaptive Windowing)
ADWIN monitors a stream $z_t \in \mathbb{R}$ (either raw observation $x_t$ or score $S_t$).
It maintains an exponential histogram of bucket sums and counts. When any split of the current window $W$ into two subwindows $W_0, W_1$ of lengths $n_0, n_1$ satisfies:
$$|\hat{\mu}_0 - \hat{\mu}_1| > \varepsilon_{\text{cut}}$$
where
$$m = \left(\frac{1}{n_0} + \frac{1}{n_1}\right)^{-1}, \quad \varepsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \ln\left(\frac{4 |W|^2}{\delta}\right)}$$
ADWIN drops the older subwindow $W_0$ and signals drift $\text{drift}_t = 1$.

### 3.2 Page-Hinkley (PH) Test
The Page-Hinkley cumulative deviation test monitors deviations from the running mean $\bar{z}_t$:
$$U_t = \sum_{i=1}^t \left( (z_i - \bar{z}_t) - \delta_{\text{PH}} \right)$$
$$m_t = \min_{1 \le k \le t} U_k$$
Drift is detected when $U_t - m_t > \lambda_{\text{PH}}$.
In two-sided mode, negative cumulative deviations are monitored symmetrically:
$$L_t = \sum_{i=1}^t \left( (\bar{z}_t - z_i) - \delta_{\text{PH}} \right)$$
$$m_t^{(L)} = \min_{1 \le k \le t} L_k$$
and drift signals if $\max(U_t - m_t, L_t - m_t^{(L)}) > \lambda_{\text{PH}}$.

---

## 4. Algorithmic Variants

### Version A — Cautious Rolling Baseline with Candidate Contamination Gate
- Maintains causal rolling window $W_t^{ref}$ and active baseline $(m_t^{base}, \text{MAD}_t^{base})$.
- Evaluates $S_t = |x_t - m_t^{base}| / (c \cdot \text{MAD}_t^{base} + \epsilon)$.
- Feeds $z_t$ into drift detector.
- Upon drift: enters pending adaptation state and collects candidate window $C_t = (x_{t+1}, \dots, x_{t+M})$.
- Candidate statistics: $m_C = \text{median}(C_t)$, $\text{MAD}_C = \text{median}(\{|x - m_C| : x \in C_t\})$.
- Internal candidate verification (location invariant):
  $$s_i^{(int)} = \frac{|x_i - m_C|}{c \cdot \text{MAD}_C + \epsilon}$$
- **Contamination Gate**: Reject $C_t$ if:
  $$\frac{1}{M} \sum_{i=1}^M \mathbf{1}\{s_i^{(int)} > \tau_{int}\} > \rho_{\text{max}} \quad \text{OR} \quad \frac{\text{MAD}_C + \epsilon}{\text{MAD}_t^{base} + \epsilon} > \kappa_{\text{max}}$$
- If accepted: $(m_{t+1}^{base}, \text{MAD}_{t+1}^{base}) \leftarrow (m_C, \text{MAD}_C)$; if rejected: discard update and retain existing baseline.

### Version B — Simplified Rolling Window + ADWIN (Ablation Baseline)
- Updates $m_t = \text{median}(W_t^{ref})$ and $\text{MAD}_t = \text{MAD}(W_t^{ref})$ at every step directly from sliding window of size $L$.
- Runs ADWIN in parallel for logging and diagnostic event telemetry.
- Serves as the ablation counterpoint to isolate the benefits of the candidate contamination gate.

### Version C — Frozen-Window Detector with Delayed Drift Adaptation
- Computes frozen baseline $(m_{\text{frozen}}, \text{MAD}_{\text{frozen}})$ from initial reference $W_0 = (x_1, \dots, x_{L_0})$.
- Scores $x_t$ continuously against the frozen baseline.
- Upon drift detection:
  - **Immediate mode**: Recomputes baseline from most recent $L_0$ samples.
  - **Delayed mode**: Collects hold buffer of $H$ steps. Evaluates whether the shift is sustained rather than an anomaly burst. If sustained and uncontaminated, re-freezes the baseline; otherwise discards.

### Version D — Clipped robust innovation tracker
- During warm-up, initialize location $m$ and scale $q$ from the first $L$ observations using median and scaled MAD.
- At $t$, score against the prior state: $r_t=x_t-m_{t-1}$, $z_t=|r_t|/q_{t-1}$.
- Combine innovation and scale channels with $S_t=\max(z_t, |r_t|/v_{t-1})$, where $v$ is a slow robust residual-scale reference.
- Alert iff $S_t>\tau$. Drift monitoring receives $x_t$ (or $z_t$ when configured).
- If $z_t\leq\tau_{clip}$, update location with clipped EW residual $m_t=m_{t-1}+\alpha\,\mathrm{clip}(r_t,\pm\tau_{clip}q)$ and update scale slowly from clipped $|r_t|$. Large isolated residuals do not update the baseline.
- On a drift trigger, reset location to the median of recent observations; scale remains slow to avoid absorbing a volatility burst immediately.
