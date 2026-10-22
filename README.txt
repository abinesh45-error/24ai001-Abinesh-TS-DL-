# LSTM Web Traffic Forecasting Project

## Run in Google Colab
1. Upload `web_traffic_lstm.py` to Google Colab, or copy its contents into a code cell.
2. Ensure TensorFlow, pandas, NumPy, Matplotlib, and scikit-learn are available (Colab usually includes them).
3. Run the script and upload the dataset CSV or ZIP when prompted.
4. The input dataset should contain a `Page` column and daily date columns, with at least 356 daily records and five usable articles.
5. The script trains an LSTM model, reports MAE/RMSE, plots actual versus predicted traffic, and forecasts the next 7 days.

## Output files
- `lstm_evaluation_metrics.csv`
- `future_traffic_forecast.csv`
- `web_traffic_lstm.keras`

Note: This script is for web-page traffic forecasting, not daily temperature prediction.
