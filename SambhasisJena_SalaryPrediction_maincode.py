import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import pickle


# Regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 📌 STEP 1: Load the CSV file
data = pd.read_csv("Sal_train.csv")

# 📌 STEP 2: Clean column names
data.columns = data.columns.str.strip().str.lower()

# 📌 STEP 3: Drop unnecessary columns if present
data.drop(columns=['id', 'crucial_code'], inplace=True, errors='ignore')

# ✅ STEP 4: Fill missing values with mean (numeric) or mode (categorical)
for col in data.columns:
    if data[col].dtype == 'object':
        mode_val = data[col].mode()[0]
        data[col] = data[col].fillna(mode_val)
    else:
        mean_val = data[col].mean()
        data[col] = data[col].fillna(mean_val)

# 📌 STEP 5: Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 📌 STEP 6: Separate features (X) and target (y)
X = data.drop(columns=['salary'])  # features
y = data['salary']  # target column

# 📌 STEP 7: Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 STEP 8: Create the models
models = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42)
}

# 📌 STEP 9: Train and evaluate each model
print("📊 Model Results:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)  # ✅ updated
    r2 = r2_score(y_test, predictions)

    print(f"{name} ➤ MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# 📌 STEP 10: Identify and save the best model (based on R²)
best_model_name = None
best_r2 = -1
best_model = None

for name, model in models.items():
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

print(f"\n✅ Best Model: {best_model_name} (R² = {best_r2:.4f})")

# 📦 Save best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# 📦 Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
