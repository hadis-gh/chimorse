# OA Weighted-Fit Comparison

This report compares the PA/OA Fourier-Morse potential fits obtained with
different alpha-weighting schemes. All fits use harmonic ceilings OA (20, 1),
with the per-orientation Morse `alpha` fitted under a given weight
distribution. Error statistics are computed from the same 243586-point
reference set (`data/PA/E_all_OA.dat`).

## Weighting schemes compared

| Tag | Scheme | Notebook |
|-----|--------|----------|
| `w_equal` | Equal weights (baseline) | `plot_oa_error.py` inline |
| `w_poisson_lam0.5` | Poisson, λ = 0.5 | `08_oa_fit_poisson_lam05.ipynb` |
| `w_poisson_lam0.8` | Poisson, λ = 0.8 | `10_oa_fit_poisson_lam08.ipynb` |
| `w_energy_lam0.25` | Energy-tied, λ = 1/4 | `11_oa_fit_energy_lam025.ipynb` |

The Poisson weight centres its mode at each orientation's $r_e$ (right-skewed,
suppressing the steep repulsive side while retaining the large-$r$ tail). The
energy-tied weight uses
$w(r)=\max(|E_{\min}|-(E(r)-E(r_e)),\epsilon)^{1/4}$ with floor
$\epsilon=10^{-4}$.

## Error statistics

| Metric | equal | Poisson λ=0.5 | Poisson λ=0.8 | Energy λ=1/4 |
|--------|------:|--------------:|--------------:|-------------:|
| MAE (all r) | 25.3484 | 18.7668 | **17.8275** | 23.6085 |
| RMSE (all r) | 41.3872 | 30.3610 | **28.7310** | 37.6839 |
| MAE (8.3–10.3 Å) | 19.1567 | 10.9137 | **10.0826** | 17.6176 |
| RMSE (8.3–10.3 Å) | 35.4359 | 20.1254 | **18.1669** | 31.8299 |
| RMSE near eq (±0.5 Å) | 16.9568 | 7.4900 | **7.3257** | 14.8783 |
| RMSE near eq (≤ +100 meV) | 7.0605 | 4.9370 | **4.7637** | 6.7009 |

Best value in each row is **bold**. All units meV.

## Observations

1. **Poisson weighting (λ=0.5 and λ=0.8) dramatically improves accuracy over
   equal weighting** on every metric — overall MAE/RMSE are reduced by roughly
   a third, and near-equilibrium RMSE by over 50%. The weighting correctly
   suppresses the poorly-modeled steep repulsive wall in favour of the
   physically-relevant well.

2. **Increasing λ from 0.5 to 0.8 improves the Poisson fit across the board.**
   A wider, less skewed Poisson distribution better captures the broad minimum
   region: near-equilibrium spatial RMSE drops 7.49 → 7.33 meV, RMSE(8.3–10.3 Å)
   20.13 → 18.17 meV, and near-equilibrium energy RMSE 4.94 → 4.76 meV.

3. **Energy-tied weighting with λ=1/4 is worse than Poisson but still better
   than equal weighting.** It improves over the baseline (near-eq spatial RMSE
   16.96 → 14.88 meV) but substantially underperforms both Poisson variants,
   suggesting the energy-tied scheme weights the well less optimally.

4. **Overall ranking by near-equilibrium spatial RMSE:**
   Poisson λ=0.8 (7.33) < Poisson λ=0.5 (7.49) < Energy λ=1/4 (14.88) < equal (16.96).

## Conclusions

- Poisson weighting is the strongest of the tested schemes; **λ=0.8 outperforms
  λ=0.5** on every reported metric, so the Poisson fits benefit from the wider
  distribution.
- The energy-tied λ=1/4 fit is only marginally better than the equal-weight
  baseline and is not competitive with the Poisson schemes for this
  interaction.

## Method note

The energy-weight formula was reverted to the earlier square-root form and the
exponent made configurable:

```
w(r) = max(|e_min| - (e - e_re), eps) ** lam     # lam default 1/2 (sqrt)
```

`lam=1/2` reproduces the original `sqrt(max(|e_min| - (e - e_re), eps))`;
`lam=1/4` (used here) yields the fourth root. Implemented in
`src/chimorse/fitting.py::energy_weights` and the `make_weight_func('energy')`
dispatcher (default changed to `1/2`).

## Artifacts

- Fit CSVs: `data/PA/df_model_OA_w_poisson_lam0.8.csv`,
  `data/PA/df_model_OA_w_energy_lam0.25.csv`
- Error plots: `Figures/oa_E_statistics_and_error_w_poisson_lam0.8.pdf`,
  `Figures/oa_E_statistics_and_error_w_energy_lam0.25.pdf` (plus bound,
  shifted, relative-error, waterfall, selected-series, and top-RMSE variants)
- Stats files: `Figures/oa_fit_error_stats_w_poisson_lam0.8.txt`,
  `Figures/oa_fit_error_stats_w_energy_lam0.25.txt`
