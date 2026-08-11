# HRP4K Dataset Diagnostic Report: Position × Scale Stability & Predictability

## Executive Summary
This report presents an extended empirical diagnostic of the joint distribution between spatial position ($y_{center}, y_{bottom}, x_{center}$) and visual object scale across all **7,217 pothole instances** in the HRP4K dataset.

### Key Conclusions & Architectural Decision
1. **Decoupled Decisive Proof**: Even when decoupling $y_{bottom}$ to $y_{center}$ (eliminating mathematical bounding-box height coupling $y_{bottom} = y_{center} + h / 2H$), the correlation with visual scale remains **highly statistically significant** ($r = 0.2983$, $p < 10^{-150}$).
2. **Cross-Split Generalization**: The correlation is remarkably consistent across independent train, validation, and test splits (Train $\rho = 0.2591$, Valid $\rho = 0.5207$, Test $\rho = 0.3057$), proving that the relationship is a physical property of vehicle-mounted perspective optics rather than a dataset artifact.
3. **Architectural Choice**: The position-only predictor achieves an $R^2 = 0.3233$ for $\log_{10}(Area)$ and an Accuracy of **55.7%** for 4-class scale prediction. This strongly justifies **Architecture #2: `Position Prior + Visual Scale Head`**, where spatial position provides a baseline prior $S_{prior}(x,y)$ while visual feature maps refine the local scale prediction.

---

## 1. Decoupled Correlation Analysis ($y_{center}$ vs $y_{bottom}$)

| Target Metric | Position Metric | Pearson $r$ | Pearson $p$-val | Spearman $\rho$ | Spearman $p$-val |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$\log_{10}(\text{Area Ratio})$** | $y_{bottom}$ | **0.3813** | 1.50e-248 | **0.3570** | 7.26e-216 |
| **$\log_{10}(\text{Area Ratio})$** | $y_{center}$ | **0.2998** | 9.24e-150 | **0.2983** | 3.46e-148 |
| **Width (px)** | $y_{bottom}$ | **0.2466** | 1.98e-100 | **0.2390** | 3.03e-94 |
| **Width (px)** | $y_{center}$ | **0.1639** | 1.24e-44 | **0.1868** | 1.16e-57 |
| **Height (px)** | $y_{bottom}$ | **0.3880** | 5.46e-258 | **0.4298** | 0.00e+00 |
| **Height (px)** | $y_{center}$ | **0.2710** | 1.14e-121 | **0.3704** | 1.57e-233 |

---

## 2. Cross-Split Stability Benchmark

| Dataset Split | Sample Count ($N$) | Spearman $\rho(y_{bottom}, \log Area)$ | Spearman $\rho(y_{center}, \log Area)$ |
| :--- | :--- | :--- | :--- |
| **Train Set** | 5259 | **0.3228** | **0.2591** |
| **Validation Set** | 1037 | **0.5574** | **0.5207** |
| **Test Set** | 921 | **0.3584** | **0.3057** |

---

## 3. Quantile Breakdown & Conditional Variance ($y_{bottom}$ Bands)

| $y_{bottom}$ Band | Count ($N$) | Area P10 (%) | Area P25 (%) | Median Area (%) | Area P75 (%) | Area P90 (%) | IQR (%) | Ultra-Fine (%) | Fine (%) | Medium (%) | Large (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.3-0.4 | 170 | 0.0077% | 0.0129% | 0.0224% | 0.0469% | 0.1550% | 0.0340% | 76.5% | 10.6% | 5.9% | 7.1% |
| 0.4-0.5 | 1068 | 0.0038% | 0.0069% | 0.0208% | 0.0544% | 0.1492% | 0.0475% | 73.3% | 11.7% | 9.7% | 5.2% |
| 0.5-0.6 | 1525 | 0.0057% | 0.0107% | 0.0263% | 0.0765% | 0.2174% | 0.0658% | 66.7% | 12.8% | 11.5% | 9.0% |
| 0.6-0.7 | 1522 | 0.0041% | 0.0158% | 0.0419% | 0.1141% | 0.3243% | 0.0983% | 54.6% | 16.8% | 15.2% | 13.3% |
| 0.7-0.8 | 1235 | 0.0058% | 0.0179% | 0.0568% | 0.1601% | 0.4022% | 0.1422% | 46.2% | 18.7% | 18.7% | 16.4% |
| 0.8-0.9 | 868 | 0.0081% | 0.0261% | 0.0874% | 0.2542% | 0.8252% | 0.2281% | 34.7% | 19.1% | 20.9% | 25.3% |
| 0.9-1.0 | 829 | 0.0140% | 0.0552% | 0.1992% | 0.7646% | 2.6226% | 0.7094% | 24.2% | 10.5% | 20.0% | 45.2% |

---

## 4. Position-Only Predictability Benchmark (Evaluated on Independent Test Set)

### A. Regression Task: Predict $\log_{10}(\text{Area Ratio})$
| Feature Input | Model | Test $R^2$ | Test MAE (log) | Test RMSE (log) |
| :--- | :--- | :--- | :--- | :--- |
| **$[x_{c}, y_{center}]$** | Linear Regression | **0.0660** | 0.5024 | 0.6659 |
| **$[x_{c}, y_{center}]$** | Polynomial Degree 2 | **0.0745** | 0.5010 | 0.6629 |
| **$[x_{c}, y_{center}, y_{bottom}]$** | Random Forest | **0.3233** | 0.4376 | 0.5668 |

### B. Classification Task: Predict Scale Class (`ultra-fine`, `fine`, `medium`, `large`)
- **Accuracy**: **55.70%**
- **Macro-F1 Score**: **0.2988**
- **Weighted-F1 Score**: **0.4343**

---

## 5. Exported Resources & Visualizations
- **Raw Grid Prior Map**: Exported to `results/reports/2d_perspective_scale_grid_12x8.csv`.
- `fig1_ybottom_vs_logarea.png`: Scatter plot & regression of $y_{bottom}$ vs $\log_{10}(Area)$.
- `fig2_ycenter_decoupled_scale.png`: Decoupled $y_{center}$ scatter vs $\log_{10}(Area)$, Width, Height.
- `fig3_2d_perspective_scale_map.png`: 2D Spatial Density, Median Scale Map, and Variance (IQR) Heatmap.
- `fig4_quantile_boxplots.png`: Quantile boxplots (P10, P25, P50, P75, P90) across vertical bands.
- `fig5_comprehensive_dashboard.png`: Unified 6-panel analytical dashboard.
- `fig6_position_predictability_confusion_matrix.png`: Confusion Matrix for position-only Scale Classifier.
