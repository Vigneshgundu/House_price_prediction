import streamlit as st
import pickle
import json
import numpy as np

model_path = r"\House_price_prediction\banglore_home_prices_model.pickle"

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load columns
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]
    locations = data_columns[3:]  # Extracting location names

# Function to predict price
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location in locations:
        loc_index = data_columns.index(location)
        x[loc_index] = 1
    return round(model.predict([x])[0], 2)

# Streamlit UI
st.title("House Price Prediction")
st.write("Enter the details below to predict the price of a house in Bangalore.")

sqft = st.number_input("Total Square Feet", min_value=100, step=10)
bhk = st.number_input("BHK", min_value=1, step=1)
bath = st.number_input("Number of Bathrooms", min_value=1, step=1)
location = st.selectbox("Location", locations)

if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"Predicted Price: ₹{price} Lakhs")
