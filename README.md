# Stock Price Prediction using LSTM 📈

A deep learning project that predicts Apple (AAPL) stock closing prices using a stacked LSTM (Long Short-Term Memory) neural network, built with `yfinance`, `Keras`, and `Plotly`.

## Overview

This project pulls ~13 years of historical AAPL price data, visualizes it as an interactive candlestick chart, and trains an LSTM-based neural network to predict the closing price from a given day's Open, High, Low, and Volume values.

## Features

- 📥 **Live data ingestion** — fetches historical OHLCV data directly from Yahoo Finance via `yfinance`
- 📊 **Interactive visualization** — candlestick chart of AAPL price history using `Plotly`
- 🔍 **Exploratory analysis** — null-value checks and feature correlation analysis against closing price
- 🧠 **Stacked LSTM model** — two LSTM layers (128 → 64 units) followed by dense layers, trained with Keras
- 🔮 **Price prediction** — predicts closing price from a custom Open/High/Low/Volume input

## Tech Stack

- Python 3.11
- [yfinance](https://pypi.org/project/yfinance/) — historical market data
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — data processing
- [Plotly](https://plotly.com/python/) — candlestick visualization
- [scikit-learn](https://scikit-learn.org/) — train/test splitting
- [Keras](https://keras.io/) — LSTM model architecture and training

## Model Architecture

```
Input (Open, High, Low, Volume)
        │
        ▼
   LSTM (128 units, return_sequences=True)
        │
        ▼
   LSTM (64 units)
        │
        ▼
   Dense (25 units)
        │
        ▼
   Dense (1 unit) → Predicted Close
```

Total parameters: ~117,619

## How It Works

1. **Data collection** — downloads the last ~5000 days of AAPL OHLCV data using `yfinance`
2. **Preprocessing** — flattens multi-index columns, selects relevant fields, and resets the index
3. **Visualization** — renders an interactive candlestick chart of price history
4. **Correlation check** — examines how strongly each feature relates to the closing price
5. **Train/test split** — 80/20 split (no shuffling, to preserve chronological order)
6. **Model training** — trains the stacked LSTM for 30 epochs using MSE loss and the Adam optimizer
7. **Prediction** — feeds a sample Open/High/Low/Volume row into the trained model to predict Close

## Getting Started

### Prerequisites

```bash
pip install pandas numpy yfinance plotly scikit-learn keras tensorflow
```

### Running the Project

1. Clone this repository
2. Open `Stock_Price_Prediction_using_LSTM.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells in order

```bash
jupyter notebook Stock_Price_Prediction_using_LSTM.ipynb
```

### Predicting on New Data

To predict the closing price for a custom day, update the input array with `[Open, High, Low, Volume]` values:

```python
features = np.array([[180.08, 184.99, 175.07, 74919600]])
prediction = model.predict(features)
print(prediction)
```

## Known Limitations & Next Steps

This is an early baseline version, and there's an important caveat worth being upfront about:

- **No real time-series sequencing yet** — the model currently predicts Close from *same-day* Open/High/Low/Volume rather than a rolling window of prior days. This means it's largely learning the (very strong) correlation between same-day price features rather than genuine temporal patterns — which is the whole point of using an LSTM in the first place.
- **Planned improvement** — reshape the input into proper sliding-window sequences (e.g., using the past *N* days of Close prices to predict the next day) so the LSTM can actually learn temporal dependencies.
- **Other ideas** — add technical indicators (moving averages, RSI), experiment with multivariate sequence inputs, and evaluate with proper time-series metrics (RMSE, MAPE) instead of relying on raw MSE loss alone.

## Disclaimer

This project is for educational purposes only and should not be used for actual financial or investment decisions. Stock price prediction is inherently uncertain, and this model does not account for market news, macroeconomic factors, or other real-world variables that influence stock prices.

## License

This project is open source and available under the [MIT License](LICENSE).
