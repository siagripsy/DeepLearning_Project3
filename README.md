# DeepLearning Project 3

This project implements **Part 2 - Option B: Time-Series Prediction** from the LSTM assignment. The goal is to forecast the **Bitcoin daily closing price** using historical data from `BTC-USD_daily.csv`.

## What Is Included

- `Project3_LSTM.ipynb`: the completed notebook for Parts 1 to 7
- `BTC-USD_daily.csv`: the daily Bitcoin dataset used for forecasting

## Task Summary

The notebook builds a multi-step forecasting pipeline based on an **LSTM** model:

- Input: a rolling window of past Bitcoin `close` prices
- Target: the next `forecast_horizon` days of `close` prices
- Forecast setting: configurable multi-day prediction
- Final visualization: the **last 30 real days** plus the **future predicted prices**

The notebook also includes:

- conceptual answers for Part 1
- dataset preparation and chronological train/validation/test splitting
- LSTM implementation in PyTorch
- model training and evaluation
- one controlled experiment
- one architecture comparison against an MLP baseline
- final forecasting visualization

## Main Hyperparameters

These values are defined inside the notebook and can be changed easily:

- `sequence_length = 60`
- `forecast_horizon = 7`
- `batch_size = 64`

The most important project-specific hyperparameter is `forecast_horizon`, which controls how many future days the model predicts.

## Environment

Run the notebook with the **`mlhub`** Jupyter kernel.

The notebook metadata has already been updated to use:

- kernel name: `mlhub`
- display name: `Python (mlhub)`

## How To Run

1. Open `Project3_LSTM.ipynb` in Jupyter or VS Code.
2. Select the `mlhub` kernel.
3. Run the notebook from top to bottom.

## Dataset Details

The dataset contains daily Bitcoin records with columns such as:

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `ticker`

This project uses the **`close`** column as the prediction target and keeps `date` for chronological splitting and plotting.

## Implemented Modeling Approach

The LSTM model:

- reads sequences shaped like `(batch_size, sequence_length, 1)`
- processes them with stacked LSTM layers
- uses the final hidden representation
- predicts the next `forecast_horizon` close prices in one forward pass

For comparison, the notebook also trains a simple **MLP baseline** on the same rolling windows.

## Reported Results

The executed notebook currently reports these main results for the default setup:

- LSTM validation RMSE: `2624.621`
- LSTM test RMSE: `16937.862`
- LSTM validation MAE: `1802.100`
- LSTM test MAE: `14016.049`

### Controlled Experiment

The controlled experiment changes **sequence length** and shows that:

- `sequence_length = 30` produced the best validation RMSE among the tested values

### Architecture Comparison

The comparison section shows that, for the current training setup:

- the **MLP baseline** achieved lower test RMSE than the LSTM

This suggests the LSTM can likely be improved further with additional hyperparameter tuning.

## Forecast Output

The notebook produces a final table and plot for the next `forecast_horizon` predicted days. In the executed version, the 7-day forecast starts after the last available date in the dataset and is appended to the final 30 real observations.

## Notes

- Scaling is fitted only on the training portion to reduce leakage.
- Data splits are chronological, not random.
- The notebook has already been executed successfully with the `mlhub` environment.
