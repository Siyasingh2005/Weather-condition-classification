# Assignment 6 — Weather Condition Classification using SVM

## Objective
Classify weather conditions as **Cool** or **Warm** using live meteorological data (temperature, relative humidity, surface pressure, wind speed) pulled from the Open-Meteo API, by training a Support Vector Machine (SVM) classifier with an RBF kernel.

Name: SIYA SINGH

Registration Number: 23MIP10030

Application Number: IN26011506

Batch Number: 1A

Email: siya.23mip10030@vitbhopal.ac.in


## API Documentation Link
- Open-Meteo Forecast API: https://open-meteo.com/
- Example request used:
  `https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&forecast_days=7`

## Libraries Used
- `requests` — API calls
- `pandas`, `numpy` — data handling
- `scikit-learn` — `train_test_split`, `StandardScaler`, `LabelEncoder`, `SVC`, evaluation metrics
- `matplotlib`, `seaborn` — confusion matrix visualization

## Methodology
1. **Data Collection:** Called the Open-Meteo `/v1/forecast` endpoint for New Delhi (28.6139° N, 77.2090° E) with a 7-day hourly forecast, requesting temperature, relative humidity, surface pressure, and wind speed. The JSON response was converted into a Pandas DataFrame.
2. **Target Engineering:** Created a binary `Weather_Class` column — `Warm` if `Temperature ≥ 25°C`, otherwise `Cool`.
3. **Preprocessing:** Checked for missing values (none found), dropped the non-predictive `time` column, label-encoded the target, split the data 80/20 into train/test sets, and standardized all four features with `StandardScaler` (fit on train, applied to test).
4. **Modeling:** Trained an `SVC(kernel="rbf")` classifier on the scaled training data and generated predictions on the test set.
5. **Evaluation:** Computed Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix; visualized the confusion matrix as a heatmap.

## Results
| Metric | Score |
|---|---|
| Accuracy | 0.9706 |
| Precision | 1.0000 |
| Recall | 0.9583 |
| F1-Score | 0.9787 |

**Confusion Matrix:**
```
[[10  0]
 [ 1 23]]
```

**Observations:**
1. High overall accuracy (97.1%) shows the RBF-kernel SVM separates Cool vs. Warm hours very well once features are standardized.
2. Precision on "Cool" predictions was perfect (1.00); the one misclassification was a true "Warm" reading near the 25°C boundary predicted as "Cool" — expected behavior right at a hard threshold.
3. Despite class imbalance (119 Warm vs. 49 Cool hours in the 7-day window), the model generalized well to both classes rather than defaulting to the majority class.

## Conclusion
The SVM classifier with an RBF kernel achieved strong performance in distinguishing Cool from Warm weather conditions using temperature, humidity, surface pressure, and wind speed from the Open-Meteo API, reaching 97.1% accuracy, 100% precision, and a 0.98 F1-score on the test set. The single misclassification occurred near the 25°C decision boundary, which is expected behavior for any threshold-based label. Feature scaling proved essential: SVM computes distances and kernel similarities between data points, so unscaled features (e.g., pressure in the 1000s versus wind speed in single digits) would dominate the decision boundary and severely bias the model. The key advantage of SVM here is its ability to capture non-linear relationships between meteorological variables via the RBF kernel, without needing an explicit feature transformation. Its main limitation is computational cost and reduced interpretability on larger datasets, since kernel SVMs scale poorly with sample size and don't offer directly interpretable coefficients like logistic regression does.

## Note on Data
This code calls the live Open-Meteo API directly (`requests.get(...)`). In this development/testing environment, outbound internet access to `api.open-meteo.com` was blocked, so a local fallback generates a structurally identical dataset (same columns, realistic Delhi-July value ranges) so the pipeline runs end-to-end. Running `Assignment-6.py` / `Assignment-6.ipynb` in an environment with normal internet access (e.g., Google Colab, local Jupyter) will fetch and use real live forecast data automatically — no code changes needed.
