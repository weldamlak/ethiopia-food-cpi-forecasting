import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load Dataset
filename = "eth_faostat_consumer_price_indices.csv"
raw_df = pd.read_csv(filename)
df = raw_df.copy()

# 2. Filter Headers & Target Indicator
if "Iso3" in df.columns:
    df = df[~df["Iso3"].astype(str).str.startswith("#")]
if "Year" in df.columns:
    df = df[~df["Year"].astype(str).str.startswith("#")]

if "Item" in df.columns:
    food_items = [i for i in df["Item"].dropna().unique() if "Food" in str(i)]
    selected_item = food_items[0] if food_items else df["Item"].unique()[0]
    df = df[df["Item"] == selected_item].copy()

# 3. Clean Numeric Types & Parse Months
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

if "Months Code" in df.columns:
    df["Months_Code_Num"] = pd.to_numeric(df["Months Code"], errors="coerce")
    df = df[
        (df["Months_Code_Num"] >= 7001) & (df["Months_Code_Num"] <= 7012)
    ].copy()
    df["Month"] = (df["Months_Code_Num"] - 7000).astype(int)

df = df.dropna(subset=["Year", "Month", "Value"]).copy()
df["Year"] = df["Year"].astype(int)
df = df.sort_values(by=["Year", "Month"]).reset_index(drop=True)

# 4. Feature Engineering (Month-over-Month Growth)
df["Time_Index"] = df["Year"] + (df["Month"] - 1) / 12.0
df["MoM_Growth_%"] = df["Value"].pct_change() * 100
df["Prev_MoM_Lag1"] = df["MoM_Growth_%"].shift(1)
df["Prev_MoM_Lag2"] = df["MoM_Growth_%"].shift(2)

# Drop missing values only in lag feature columns
model_df = df.dropna(
    subset=["MoM_Growth_%", "Prev_MoM_Lag1", "Prev_MoM_Lag2"]
).copy()

print(f"✅ Cleaned records: {len(df)}")
print(f"📈 Machine Learning samples ready: {len(model_df)}")

# 5. Train Model & Split Data
X = model_df[["Month", "Prev_MoM_Lag1", "Prev_MoM_Lag2"]]
y = model_df["MoM_Growth_%"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate reconstructed CPI
y_pred_mom = model.predict(X_test)
test_actual_cpi = model_df["Value"].iloc[-len(y_test) :]

reconstructed_cpi = []
for i in range(len(test_actual_cpi)):
    prev = (
        model_df["Value"].iloc[-len(y_test) - 1]
        if i == 0
        else reconstructed_cpi[-1]
    )
    reconstructed_cpi.append(prev * (1 + y_pred_mom[i] / 100.0))

mae = mean_absolute_error(test_actual_cpi, reconstructed_cpi)

print("\n" + "=" * 55)
print("             MODEL EVALUATION RESULTS")
print("=" * 55)
print(f"Mean Absolute Error (MAE): {mae:.2f} CPI points")

# 6. Forecast Next Month
last_year = int(model_df["Year"].iloc[-1])
last_month = int(model_df["Month"].iloc[-1])
last_actual_cpi = model_df["Value"].iloc[-1]
last_mom = model_df["MoM_Growth_%"].iloc[-1]
prev_mom = model_df["MoM_Growth_%"].iloc[-2]

next_month = 1 if last_month == 12 else last_month + 1
next_year = last_year + 1 if last_month == 12 else last_year

next_features = pd.DataFrame(
    [
        {
            "Month": next_month,
            "Prev_MoM_Lag1": last_mom,
            "Prev_MoM_Lag2": prev_mom,
        }
    ]
)

pred_next_mom = model.predict(next_features)[0]
forecasted_cpi = last_actual_cpi * (1 + pred_next_mom / 100.0)

print("\n" + "=" * 55)
print(f"     FORECAST FOR NEXT PERIOD ({next_month}/{next_year})")
print("=" * 55)
print(f"Latest Recorded CPI ({last_month}/{last_year}):  {last_actual_cpi:.2f}")
print(f"Predicted MoM Inflation Growth: {pred_next_mom:+.2f}%")
print(f"Predicted CPI for ({next_month}/{next_year}):    {forecasted_cpi:.2f}")
print("=" * 55)