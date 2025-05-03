import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 1. Generate 300 inlier points around a single center
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6, random_state=42)

# 2. Create 20 random outlier points uniformly distributed
outliers = np.random.uniform(low=-10, high=10, size=(20, 2))

# 3. Combine inliers and outliers into one dataset
X_full = np.vstack([X, outliers])

# 4. Plot the combined dataset
plt.scatter(X_full[:, 0], X_full[:, 1])
plt.title("Synthetic Dataset with Outliers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
from scipy.stats import zscore

# 5. Calculate Z-scores
z_scores = np.abs(zscore(X_full))

# 6. Identify outliers (Z-score > 3)
is_outlier = (z_scores > 3).any(axis=1)

# 7. Plot the results
plt.scatter(X_full[:, 0], X_full[:, 1], c=~is_outlier, cmap='coolwarm')
plt.title("Outlier Detection using Z-Score")
plt.show()
from sklearn.ensemble import IsolationForest

# 8. Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.06, random_state=42)
preds = iso_forest.fit_predict(X_full)

# 9. Mark outliers (–1 means outlier)
iso_outliers = preds == -1

# 10. Plot Isolation Forest results
plt.figure()
plt.scatter(X_full[:, 0], X_full[:, 1], c=~iso_outliers, cmap='coolwarm')
plt.title("Outlier Detection using Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
from sklearn.cluster import DBSCAN

# 11. Fit DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_full)

# 12. Mark outliers (label == -1)
dbscan_outliers = labels == -1

# 13. Plot DBSCAN results
plt.figure()
plt.scatter(X_full[:, 0], X_full[:, 1], c=~dbscan_outliers, cmap='coolwarm')
plt.title("Outlier Detection using DBSCAN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
# 14. Evaluate and compare the number of outliers detected by each method
print(f"Outliers detected by Z-Score: {np.sum(is_outlier)}")
print(f"Outliers detected by Isolation Forest: {np.sum(iso_outliers)}")
print(f"Outliers detected by DBSCAN: {np.sum(dbscan_outliers)}")
from sklearn.metrics import classification_report

# Ground‑truth labels: 1 for true outlier, 0 for inlier
true_labels = np.zeros(len(X_full), dtype=int)
true_labels[-20:] = 1

# Prepare predicted label arrays (1 = outlier, 0 = inlier)
pred_z     = is_outlier.astype(int)
pred_iso   = iso_outliers.astype(int)
pred_db    = dbscan_outliers.astype(int)

# Print reports
print("\nZ‑Score Classification Report:")
print(classification_report(true_labels, pred_z, zero_division=0))

print("Isolation Forest Classification Report:")
print(classification_report(true_labels, pred_iso, zero_division=0))

print("DBSCAN Classification Report:")
print(classification_report(true_labels, pred_db, zero_division=0))

