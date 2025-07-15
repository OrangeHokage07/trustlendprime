import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("urban_credit_5000.csv")
df.drop(columns=["user_id", "name", "phone_number", "age"], inplace=True)

target = "target_credit_score"
X = df.drop(columns=[target])
y = df[target]

categorical = ["employment_status"]
numerical = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), numerical),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train_transformed, y_train)

y_pred = model.predict(X_test_transformed)
y_pred = np.clip(y_pred, 300, 850)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Final RMSE: {rmse:.2f}")
print(f"✅ Final R² Score: {r2:.2f}")

explainer = shap.Explainer(model, X_train_transformed)
shap_values = explainer(X_test_transformed)

feature_names = preprocessor.get_feature_names_out()

shap.summary_plot(
    shap_values,
    features=X_test_transformed,
    feature_names=feature_names,
    show=False
)
plt.savefig("shap_summary_plot.png")

bar_explanation = shap.Explanation(
    values=shap_values.values.mean(0),
    base_values=shap_values.base_values.mean(0),
    data=X_test_transformed[0],
    feature_names=feature_names
)

shap.plots.bar(bar_explanation, show=False)
plt.savefig("shap_bar_plot.png")

joblib.dump(model, "credit_model_xgb.pkl")
joblib.dump(preprocessor, "credit_preprocessor.pkl")
print("🎯 Model and preprocessor saved!")
