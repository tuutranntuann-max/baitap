# =========================
# IMPORT
# =========================
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("housing.csv")

# =========================
# DEFINE COLUMNS
# =========================
target = "price"

num_cols = df.select_dtypes(include=np.number).columns.drop(target)
cat_cols = df.select_dtypes(include="object").columns

text_col = None
if "description" in df.columns:
    text_col = "description"
    cat_cols = cat_cols.drop(text_col)

# =========================
# PIPELINE COMPONENTS
# =========================

# Numerical
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("power", PowerTransformer())
])

# Categorical
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

# Text
if text_col:
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50))
    ])

# =========================
# COLUMN TRANSFORMER
# =========================
transformers = [
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
]

if text_col:
    transformers.append(("text", text_pipeline, text_col))

preprocessor = ColumnTransformer(transformers)

# =========================
# BÀI 1: BUILD PIPELINE
# =========================
model = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor())
])

# Smoke test
print("\n=== SMOKE TEST ===")
sample = df.head(10)
model.fit(sample.drop(target, axis=1), sample[target])
print("Pipeline chạy OK")

# =========================
# BÀI 2: TEST PIPELINE
# =========================

print("\n=== TEST CASES ===")

# Case 1: Missing
df_missing = df.copy()
df_missing.iloc[0,0] = np.nan

# Case 2: Unseen category
df_unseen = df.copy()
if len(cat_cols) > 0:
    df_unseen.iloc[0][cat_cols[0]] = "NEW_CATEGORY"

# Case 3: Wrong format
df_wrong = df.copy()
df_wrong.iloc[0,0] = "error"

test_sets = [df, df_missing, df_unseen, df_wrong]

for i, data in enumerate(test_sets):
    try:
        X = data.drop(target, axis=1)
        y = data[target]
        model.fit(X, y)
        pred = model.predict(X)

        print(f"\nTest {i+1}: OK - Shape:", pred.shape)
    except Exception as e:
        print(f"\nTest {i+1}: ERROR -", e)

# =========================
# BÀI 3: MODEL + CV
# =========================
X = df.drop(target, axis=1)
y = df[target]

models = {
    "Linear": LinearRegression(),
    "RF": RandomForestRegressor()
}

for name, m in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", m)
    ])

    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse = np.sqrt(-scores)

    print(f"\n{name}")
    print("RMSE:", rmse.mean())
    print("STD:", rmse.std())

# =========================
# BÀI 4: SAVE + PREDICT
# =========================

# Train final model
model.fit(X, y)

# Save
joblib.dump(model, "house_model.pkl")

# Load
loaded_model = joblib.load("house_model.pkl")

# Predict function
def predict_price(new_data):
    new_df = pd.DataFrame(new_data)
    return loaded_model.predict(new_df)

# Test predict
test_input = X.iloc[:1].to_dict(orient="records")
print("\nPrediction:", predict_price(test_input))
