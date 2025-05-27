import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the trained model
model = joblib.load('../models/random_forest_model.joblib')

# 2. Load product names (only 'name' column)
df = pd.read_csv('../data/cleaned_data_final.csv')
product_names = df['name'].dropna().unique()

# 3. Define prediction function
def predict_discount_price(ratings, no_of_ratings, actual_price, discount_percent,
                            name_length, keyword_boat, keyword_amazon):
    input_features = np.array([[ratings, no_of_ratings, actual_price, discount_percent, 
                                name_length, keyword_boat, keyword_amazon]])
    prediction = model.predict(input_features)
    return prediction[0]

# 4. Build the Streamlit UI
st.set_page_config(page_title="Discount Price Predictor", page_icon="📉")
st.title("📉 Discount Price Prediction App")
st.markdown("This tool estimates the **discounted price** of an electronic product based on various features using a machine learning model.")

# 5. Product selector
st.subheader("Product Selection")

selected_product = st.selectbox(
    "🔎 Search or Select a Product Name",
    options=sorted(product_names),
    index=0
)

# Auto-compute name-based features
name_length = len(selected_product)
keyword_boat = int("boAt" in selected_product or "boat" in selected_product.lower())
keyword_amazon = int("amazon" in selected_product.lower())

# 6. User inputs
st.subheader("Product Characteristics")

ratings = st.slider("⭐ Product Rating (1.0 to 5.0)", 1.0, 5.0, 4.0, 0.1)
no_of_ratings = st.number_input("👥 Number of Ratings", min_value=0, value=1000, step=100)
actual_price = st.number_input("💰 Actual Price (₹)", min_value=0.0, value=10000.0, step=100.0)
discount_percent = st.number_input("🔻 Discount Percentage (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

# Display auto-detected name features
st.markdown("### 🔍 Name-based Features")
st.write(f"🔠 Name Length: `{name_length}`")
st.write(f"🎧 Contains 'boAt': `{bool(keyword_boat)}`")
st.write(f"📦 Contains 'Amazon': `{bool(keyword_amazon)}`")

# 7. Make prediction
if st.button("Predict Discounted Price"):
    predicted_price = predict_discount_price(ratings, no_of_ratings, actual_price, discount_percent,
                                             name_length, keyword_boat, keyword_amazon)
    st.success(f"🧮 Estimated Discounted Price: ₹{predicted_price:.2f}")
