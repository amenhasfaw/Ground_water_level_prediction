import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle

def build_model(dataset_path):
    # Load dataset
    dataset_cleaned = pd.read_csv('dataset_final')
    
    # Splitting the data into input features (X) and target variable (y)
    X = dataset_cleaned[['Precipitation', 'Humidity', 'Temperature']]  # Input features
    y = dataset_cleaned['Ground_water_level']  # Target variable
    
    # Splitting the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initializing the Gradient Boosting Regressor model
    model = GradientBoostingRegressor(random_state=42)
    
    # Training the model on the training data
    model.fit(X_train, y_train)
    
    return model

def load_model(model_path):
    # Load the trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
