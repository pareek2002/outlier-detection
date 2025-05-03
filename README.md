# Outlier Detection Project

**Dataset:** 300 inliers + 20 injected outliers (total 320 points)

## Methods and Performance

| Method             | Precision (outlier) | Recall (outlier) | F1‑Score (outlier) |
|--------------------|---------------------|------------------|--------------------|
| Z‑Score            | 1.00                | 0.70             | 0.82               |
| Isolation Forest   | 0.95                | 0.95             | 0.95               |
| DBSCAN             | 0.87                | 1.00             | 0.93               |

## Conclusions

- **Isolation Forest** gave the best balance (F1 = 0.95).  
- **DBSCAN** caught all true outliers (recall = 1.00) but included some false positives.  
- **Z‑Score** was precise but missed some outliers (recall = 0.70).

## Next Steps

- Tune hyperparameters (e.g., `eps` in DBSCAN, `contamination` in Isolation Forest).  
- Try other methods (One‑Class SVM, Local Outlier Factor).  
- Apply to real‑world datasets and compare.
