# Predictive Maintenance: Machine Failure Prediction

Predicts whether an industrial machine will fail from sensor readings (Python, scikit-learn).

## Dataset
- AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0), 10,000 rows, synthetic.
- Download: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Place `ai4i2020.csv` next to the notebook.

## Method
- Removed ID columns and the failure-reason columns (TWF, HDF, PWF, OSF, RNF) to avoid data leakage.
- Stratified 80/20 split (only ~3.4% of machines fail).
- Compared: baseline, Logistic Regression, Random Forest.
- Chose the decision threshold with 5-fold cross-validation on training data only.
- Added engineered features (power, temperature difference, wear x torque).

## Results (test set, 68 failures)
| Model | Failures caught | False alarms | F1 (failure) |
|---|---|---|---|
| Always "OK" baseline | 0 | 0 | 0.00 |
| Logistic Regression | 56 | 339 | 0.24 |
| Random Forest (0.5) | 32 | 2 | 0.63 |
| Random Forest (tuned, 0.25) | 50 | 21 | 0.72 |
| RF + engineered features (0.35) | 56 | 4 | 0.88 |

## Key learnings
- A model that always predicts "OK" gets 96.6% accuracy, so I used precision, recall and F1 for the failure class.
- Torque, rotational speed and tool wear are the most important features.
- The engineered features mirror the failure rules described for this synthetic dataset (power, heat dissipation, overstrain), so the 0.88 is likely optimistic and may not transfer to real machines.
- The test set has only 68 failures, so numbers can vary.

## Run
Install the dependencies, then open the notebook `predictive_maintenance.ipynb`:

    pip install -r requirements.txt

## Credit
Dataset: UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset (CC BY 4.0).
