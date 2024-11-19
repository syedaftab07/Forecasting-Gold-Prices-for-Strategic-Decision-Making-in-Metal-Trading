import streamlit as st
import pickle
import numpy as np

# Load the stacked model and scaler
with open('stacked_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app title
st.title("Gold Price Forecasting")

# User input for lagged prices
st.header("Enter Last 5 Days Lagged Prices")
lag1 = st.number_input("Lag 1 Price", min_value=0.0, step=0.01)
lag2 = st.number_input("Lag 2 Price", min_value=0.0, step=0.01)
lag3 = st.number_input("Lag 3 Price", min_value=0.0, step=0.01)
lag4 = st.number_input("Lag 4 Price", min_value=0.0, step=0.01)
lag5 = st.number_input("Lag 5 Price", min_value=0.0, step=0.01)

# Button to make prediction
if st.button("Forecast Gold Price"):
    # Prepare input features
    input_features = np.array([[lag1, lag2, lag3, lag4, lag5]])
    
    # Scale the features
    input_scaled = scaler.transform(input_features)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Display the prediction
    st.success(f"Predicted Gold Price for Next Day: ${prediction[0]:.2f}")

# Extend functionality to forecast for the next 30 days
st.markdown("### Extend Forecasting")
if st.button("Forecast Next 30 Days"):
    # Prepare input features for future predictions
    last_prices = [lag1, lag2, lag3, lag4, lag5]
    future_predictions = []

    for _ in range(30):
        input_scaled = scaler.transform([last_prices])
        next_price = model.predict(input_scaled)[0]
        future_predictions.append(next_price)
        # Update last_prices for the next iteration
        last_prices = last_prices[1:] + [next_price]

    # Display future predictions
    st.subheader("Predicted Gold Prices for Next 30 Days")
    st.line_chart(future_predictions)
