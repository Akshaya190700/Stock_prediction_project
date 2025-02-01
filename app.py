import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os



app = Flask(__name__)

# Load the model (make sure your model is in the correct path)
model = load_model('stock_model.keras')




@app.route('/', methods=['GET', 'POST'])
def stock_prediction_dashboard():
    if request.method == 'POST':
        stock_ticker = request.form.get('stock', 'NIFTYBEES.NS')  # Default stock if none is entered
        
        # Download stock data
        start_date = dt.datetime(2002, 1, 1)
        end_date = dt.datetime(2025, 1, 31)
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
        
        # Descriptive Statistics
        descriptive_statistics = stock_data.describe()

        # Calculate EMA (100 & 200)
        ema_100 = stock_data.Close.ewm(span=100, adjust=False).mean()
        ema_200 = stock_data.Close.ewm(span=200, adjust=False).mean()
        
        # Prepare data for model input
        x_train = pd.DataFrame(stock_data['Close'][:int(len(stock_data)*0.80)])
        x_test = pd.DataFrame(stock_data['Close'][int(len(stock_data)*0.80):])
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_train)
        
        # Prepare input data for prediction
        past_100 = x_train.tail(100)
        final_data = pd.concat([past_100, x_train], ignore_index=True)
        scaled_input_data = scaler.fit_transform(final_data)
        
        X_test_data = []
        y_test_data = []
        for i in range(100, scaled_input_data.shape[0]):
            X_test_data.append(scaled_input_data[i-100:i])
            y_test_data.append(scaled_input_data[i, 0])
        X_test_data, y_test_data = np.array(X_test_data), np.array(y_test_data)
        
        # Make Predictions
        predicted_prices = model.predict(X_test_data)
        inverse_scaling_factor = 1 / scaler.scale_[0]
        predicted_prices = predicted_prices * inverse_scaling_factor
        y_test_data = y_test_data * inverse_scaling_factor
        
        # Generate Charts for Visualization
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(stock_data.Close, 'y', label='Actual Price')
        ax1.plot(ema_100, 'g', label='EMA 100')
        ax1.plot(ema_200, 'r', label='EMA 200')
        ax1.set_title(f"Stock Price vs Time (100 & 200 Days EMA) for {stock_ticker}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = os.path.join('static', 'ema200.png')
        fig1.savefig(ema_chart_path)
        plt.close(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(y_test_data, 'g', label="Original Price", linewidth=1)
        ax2.plot(predicted_prices, 'r', label="Predicted Price", linewidth=1)
        ax2.set_title(f"Prediction vs Actual Trend for {stock_ticker}")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        prediction_chart_path = os.path.join('static', 'stock_pred.png')
        fig2.savefig(prediction_chart_path)
        plt.close(fig2)
        
        # Save CSV
        csv_path = os.path.join('static', f'{stock_ticker}_dataset.csv')
        stock_data.to_csv(csv_path)
        
        return render_template('index1.html', 
                               plot_path_ema_100_200=ema_chart_path, 
                               plot_path_prediction=prediction_chart_path, 
                               data_desc=descriptive_statistics.to_html(classes='table table-bordered'),
                               dataset_link=csv_path)
    return render_template('index1.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)