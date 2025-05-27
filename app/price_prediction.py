import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import zipfile

# 1. Define paths for model zip and extracted model
zip_path = 'models/random_forest_model.joblib.zip'
model_path = 'models/random_forest_model.joblib'

# 2. Unzip the model if not already extracted
if not os.path.exists(model_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('models/')

# 3. Load the trained model
model = joblib.load(model_path)

# 4. Load product names from dataset (only 'name' column)
df = pd.read_csv('../data/cleaned_data_final.csv')
product_names = df['name'].dropna().unique()

# 5. Define prediction function
def predict_discount_price(ratings, no_of_ratings, actual_price, discount_percent,
                           name_length, keyword_boat, keyword_amazon):
    input_features = np.array([[ratings, no_of_ratings, actual_price, discount_percent,
                                name_length, keyword_boat, keyword_amazon]])
    prediction = model.predict(input_features)
    return prediction[0]

# 6. Build Streamlit UI
st.set_page_config(page_title="Discount Price Predictor", page_icon="📉")
st.title("📉 Discount Price Prediction App")
st.markdown(
    "This app estimates the **discounted price** of an electronic product "
    "based on various features using a trained machine learning model."
)

# 7. Product selection with search box (selectbox allows typing)
st.subheader("Product Selection")
selected_product = st.selectbox(
    "🔎 Search or Select a Product Name",
    options=sorted(product_names),
    index=0
)

# 8. Auto-generate name-based features
name_length = len(selected_product)
keyword_boat = int("boAt" in selected_product or "boat" in selected_product.lower())
keyword_amazon = int("amazon" in selected_product.lower())

# 9. User inputs for other product features
st.subheader("Product Characteristics")
ratings = st.slider("⭐ Product Rating (1.0 to 5.0)", 1.0, 5.0, 4.0, 0.1)
no_of_ratings = st.number_input("👥 Number of Ratings", min_value=0, value=1000, step=100)
actual_price = st.number_input("💰 Actual Price (₹)", min_value=0.0, value=10000.0, step=100.0)
discount_percent = st.number_input("🔻 Discount Percentage (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

# 10. Display the auto-detected name features for info
st.markdown("### 🔍 Name-based Features")
st.write(f"🔠 Name Length: `{name_length}`")
st.write(f"🎧 Contains 'boAt': `{bool(keyword_boat)}`")
st.write(f"📦 Contains 'Amazon': `{bool(keyword_amazon)}`")

# 11. Predict discounted price when button is clicked
if st.button("Predict Discounted Price"):
    predicted_price = predict_discount_price(
        ratings, no_of_ratings, actual_price, discount_percent,
        name_length, keyword_boat, keyword_amazon
    )
    st.success(f"🧮 Estimated Discounted Price: ₹{predicted_price:.2f}")
