# Optimizers for Regression — GD, SGD, Mini-Batch GD, RMSprop, Adam

Five Jupyter notebooks that implement and compare five optimization algorithms on the same **regression** problem.
Every optimizer is written **from scratch in NumPy** (Linear Regression + MSE loss), so each update rule is visible line by line.

## Dataset — California Housing

- 20,640 samples, 8 numeric features (median income, house age, rooms, location, ...)
- Target: median house value (in units of $100,000)
- Loaded with `sklearn.datasets.fetch_california_housing` (no manual download needed)

Preprocessing: 80/20 train/test split → outlier clipping (1st–99th percentile of train) → standardization.

## Notebooks

| # | Notebook | Optimizer |
|---|---|---|
| 1 | [01_Batch_Gradient_Descent.ipynb](01_Batch_Gradient_Descent.ipynb) | Batch Gradient Descent |
| 2 | [02_Stochastic_Gradient_Descent.ipynb](02_Stochastic_Gradient_Descent.ipynb) | Stochastic Gradient Descent (SGD) |
| 3 | [03_MiniBatch_Gradient_Descent.ipynb](03_MiniBatch_Gradient_Descent.ipynb) | Mini-Batch Gradient Descent |
| 4 | [04_RMSprop.ipynb](04_RMSprop.ipynb) | RMSprop |
| 5 | [05_Adam.ipynb](05_Adam.ipynb) | Adam |

Each notebook contains: dataset + EDA, preprocessing, the model and gradients, a closed-form baseline,
the optimizer implementation, loss curves, predicted-vs-actual plot, weight comparison, and a learning-rate experiment.

## Results (test set)

| Optimizer | Settings | Test MSE | Test R² |
|---|---|---|---|
| Closed form (optimal) | — | 0.4492 | 0.6572 |
| Batch GD | lr=0.1, 500 epochs | 0.4493 | 0.6572 |
| SGD | lr=0.001 + decay, 20 epochs | 0.4515 | 0.6554 |
| Mini-Batch GD | lr=0.01, batch 64, 50 epochs | 0.4495 | 0.6570 |
| RMSprop | lr=0.001, batch 64, 50 epochs | 0.4493 | 0.6571 |
| Adam | lr=0.001, batch 64, 50 epochs | 0.4494 | 0.6570 |

All five optimizers reach (almost) the optimal solution; they differ in speed, number of updates, and how noisy the loss curve is.

## How to run

**Google Colab:** File → Upload notebook (or open from GitHub), then Runtime → Run all.

**Kaggle:** New Notebook → File → Import Notebook, turn **Settings → Internet → On** (needed to download the dataset), then Run All.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook
```
