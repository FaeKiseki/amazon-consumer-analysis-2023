# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import joblib

# 2. Load cleaned data
file_path = "data/cleaned_data_final.csv"
df = pd.read_csv(file_path)

# 3. Create new features
df['name_length'] = df['name'].apply(len)
df['keyword_boat'] = df['name'].str.contains('boAt', case=False).astype(int)
df['keyword_amazon'] = df['name'].str.contains('Amazon', case=False).astype(int)

# 4. Define features
features = ['actual_price', 'discount_price', 'discount_percent', 'name_length', 'keyword_boat', 'keyword_amazon']

# 5. Prepare data for popularity prediction (regression)
X_pop = df[features]
y_pop = df['no_of_ratings']

# 6. Prepare data for ratings prediction (regression)
X_rating = df[features]
y_rating = df['ratings']

# 7. Prepare data for product quality classification (qualitative)
def get_quality(rating):
    if rating >= 4:
        return 'good'
    elif rating == 3:
        return 'average'
    else:
        return 'bad'

df['product_quality'] = df['ratings'].apply(get_quality)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['quality_encoded'] = le.fit_transform(df['product_quality'])

X_qual = df[features]
y_qual = df['quality_encoded']

# 8. Split data into train and test sets (80/20 split)
X_train_pop, X_test_pop, y_train_pop, y_test_pop = train_test_split(X_pop, y_pop, test_size=0.2, random_state=42)
X_train_rat, X_test_rat, y_train_rat, y_test_rat = train_test_split(X_rating, y_rating, test_size=0.2, random_state=42)
X_train_qual, X_test_qual, y_train_qual, y_test_qual = train_test_split(X_qual, y_qual, test_size=0.2, random_state=42)

# 9. Train RandomForestRegressor for popularity prediction
model_pop = RandomForestRegressor(n_estimators=100, random_state=42)
model_pop.fit(X_train_pop, y_train_pop)

# 10. Predict and evaluate popularity
y_pred_pop = model_pop.predict(X_test_pop)
rmse_pop = np.sqrt(mean_squared_error(y_test_pop, y_pred_pop))
mae_pop = mean_absolute_error(y_test_pop, y_pred_pop)
print(f"Popularity Prediction RMSE: {rmse_pop:.2f}")
print(f"Popularity Prediction MAE: {mae_pop:.2f}")

# 11. Train GradientBoostingRegressor for ratings prediction
model_rat = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_rat.fit(X_train_rat, y_train_rat)

# 12. Predict and evaluate ratings
y_pred_rat = model_rat.predict(X_test_rat)
rmse_rat = np.sqrt(mean_squared_error(y_test_rat, y_pred_rat))
mae_rat = mean_absolute_error(y_test_rat, y_pred_rat)
print(f"Ratings Prediction RMSE: {rmse_rat:.3f}")
print(f"Ratings Prediction MAE: {mae_rat:.3f}")

# 13. Train RandomForestClassifier for product quality classification
model_qual = RandomForestClassifier(n_estimators=100, random_state=42)
model_qual.fit(X_train_qual, y_train_qual)

# 14. Predict and evaluate product quality classification
y_pred_qual = model_qual.predict(X_test_qual)
accuracy_qual = accuracy_score(y_test_qual, y_pred_qual)
print(f"Product Quality Classification Accuracy: {accuracy_qual:.3f}")
print("Classification Report:")
print(classification_report(y_test_qual, y_pred_qual, target_names=le.classes_))

# 15. Save models to /models/ directory
import os
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model_pop, os.path.join(model_dir, "popularity_model.joblib"))
joblib.dump(model_rat, os.path.join(model_dir, "ratings_model.joblib"))
joblib.dump(model_qual, os.path.join(model_dir, "quality_model.joblib"))

print("Models saved successfully.")
