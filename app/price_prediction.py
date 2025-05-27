import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import urllib.request

# 1. Define the OneDrive direct download URL
MODEL_URL = "https://direct_download_link_from_generator"  # 🔁 Replace this with the actual link
MODEL_PATH = "models/random_forest_model.joblib"
DATA_PATH = "data/cleaned_data_final.csv"

# 2. Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 3. Download model if missing
if not os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Downloading machine learning model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("✅ Model downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Could not download the model: {e}")
        st.stop()

# 4. Load model
model = joblib.load(MODEL_PATH)

# 5. Load data
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("❌ Data file not found. Please make sure `data/cleaned_data_final.csv` is included in the GitHub repo.")
    st.stop()

product_names = df['name'].dropna().unique()

# 6. Define prediction function
def predict_discount_price(ratings, no_of_ratings, actual_price, discount_percent,
                           name_length, keyword_boat, keyword_amazon):
    input_features = np.array([[ratings, no_of_ratings, actual_price, discount_percent,
                                name_length, keyword_boat, keyword_amazon]])
    prediction = model.predict(input_features)
    return prediction[0]

# 7. Build Streamlit UI
st.set_page_config(page_title="Discount Price Predictor", page_icon="📉")
st.title("📉 Discount Price Prediction App")
st.markdown("This tool estimates the **discounted price** of an electronic product using a machine learning model.")

# 8. Product selector
st.subheader("Product Selection")
selected_product = st.selectbox("🔎 Search or Select a Product Name", options=sorted(product_names), index=0)

# 9. Auto-compute name-based features
name_length = len(selected_product)
keyword_boat = int("boAt" in selected_product or "boat" in selected_product.lower())
keyword_amazon = int("amazon" in selected_product.lower())

# 10. User inputs
st.subheader("Product Characteristics")
ratings = st.slider("⭐ Product Rating (1.0 to 5.0)", 1.0, 5.0, 4.0, 0.1)
no_of_ratings = st.number_input("👥 Number of Ratings", min_value=0, value=1000, step=100)
actual_price = st.number_input("💰 Actual Price (₹)", min_value=0.0, value=10000.0, step=100.0)
discount_percent = st.number_input("🔻 Discount Percentage (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

# 11. Display auto-detected features
st.markdown("### 🔍 Name-based Features")
st.write(f"🔠 Name Length: `{name_length}`")
st.write(f"🎧 Contains 'boAt': `{bool(keyword_boat)}`")
st.write(f"📦 Contains 'Amazon': `{bool(keyword_amazon)}`")

# 12. Prediction
if st.button("Predict Discounted Price"):
    predicted_price = predict_discount_price(ratings, no_of_ratings, actual_price, discount_percent,
                                             name_length, keyword_boat, keyword_amazon)
    st.success(f"🧮 Estimated Discounted Price: ₹{predicted_price:.2f}")
