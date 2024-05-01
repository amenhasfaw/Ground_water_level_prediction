import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model_file_path = 'gradient_boosting_regressor_model.pkl'
with open(model_file_path, 'rb') as file:
    model = joblib.load('gradient_boosting_regressor_model.joblib')

# Function to predict Ground Water Level
def predict_ground_water_level(precipitation, humidity, temperature):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({'Precipitation': [precipitation],
                               'Humidity': [humidity],
                               'Temperature': [temperature]})
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app layout
st.title('Ground Water Level Prediction')

# Input sections
precipitation = st.number_input('Precipitation (mm)', min_value=0.0, step=0.1)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, step=1.0)
temperature = st.number_input('Temperature (°C)', min_value=-50.0, max_value=50.0, step=0.1)

# Output section
if st.button('Predict'):
    prediction = predict_ground_water_level(precipitation, humidity, temperature)
    st.write(f'Predicted Ground Water Level: {prediction:.2f}')
