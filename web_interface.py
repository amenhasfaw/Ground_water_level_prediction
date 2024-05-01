# web_interface.py
import streamlit as st
import pandas as pd
import model  # Import the model module

# Streamlit app layout
st.title('Ground Water Level Prediction')

# Input sections
precipitation = st.number_input('Precipitation (mm)', min_value=0.0, step=0.1)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, step=1.0)
temperature = st.number_input('Temperature (°C)', min_value=-50.0, max_value=50.0, step=0.1)

# Output section
if st.button('Predict'):
    # Load the model
    model_path = 'gradient_boosting_regressor_model.pkl'  # Specify the model path
    model = model.load_model(model_path)
    
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({'Precipitation': [precipitation],
                               'Humidity': [humidity],
                               'Temperature': [temperature]})
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction
    st.write(f'Predicted Ground Water Level: {prediction[0]:.2f}')
