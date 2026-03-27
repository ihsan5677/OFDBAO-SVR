import OFDBAO_SVR as ofdbao
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

POP_SIZE = 15
MAX_ITER = 100
lower_bound = [1, 0.0001, 0.001]
upper_bound = [100, 10, 1]
dim = 3

window_size = 4320   # 180 days (hourly) # 720 for 01 month
horizon = 24      # 24-hours ahead
step = 1

y_true, y_pred = [], []

# --- Helper Functions ---
def load_and_prepare_energy(file_path, label):
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    if 'datetime' not in df.columns:
        df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
    # Force consistent datetime format and UTC
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', utc=True)
    df.dropna(subset=['datetime'], inplace=True)
    usage_col = None
    for c in df.columns:
        if 'usage' in c.lower() or '(kw)' in c.lower():
            usage_col = c
            break
    if usage_col is None:
        raise ValueError(f"No usage column found in {file_path}")
    df = df[['datetime', usage_col]].rename(columns={usage_col: f'usage_{label}'})
    return df

def load_weather(file_path):
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'datetime' not in df.columns:
        df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', utc=True)
    df.dropna(subset=['datetime'], inplace=True)
    return df
# --- Load Datasets ---
weather = load_weather("weather_dataset/Islamabad.csv")
energy_files = [
    ("energy_datasets/islamabad_House41.csv", "H1"),
    ("energy_datasets/islamabad_House42.csv", "H2"),
    ("energy_datasets/islamabad_House43.csv", "H3"),
    ("energy_datasets/islamabad_House44.csv", "H4"),
    ("energy_datasets/islamabad_House45.csv", "H5"),
    ("energy_datasets/islamabad_House46.csv", "H6"),
   ("energy_datasets/islamabad_House47.csv", "H7"),
    ("energy_datasets/islamabad_House48.csv", "H8"),
    ("energy_datasets/islamabad_House49.csv", "H9"),
    ("energy_datasets/islamabad_House50.csv", "H10")]

# --- Merge all energy datasets by datetime ---
merged_energy = None
for file, label in energy_files:
    e = load_and_prepare_energy(file, label)
    merged_energy = e if merged_energy is None else pd.merge(merged_energy, e, on='datetime', how='outer')
print("1:",merged_energy.shape)
# Round timestamps to hour to avoid merge mismatches
merged_energy['datetime'] = merged_energy['datetime'].dt.floor('H')
merged_energy = merged_energy.sort_values('datetime').set_index('datetime')
merged_energy = merged_energy.interpolate(method='time', limit_direction='both').reset_index()
# Compute total usage
merged_energy['TotalUsage'] = merged_energy.filter(like='usage_').sum(axis=1)
# Resample hourly
merged_energy = merged_energy.set_index('datetime').resample('1H').sum().reset_index()
# Filter by date range
start_date, end_date = '2023-11-01', '2024-10-30'
merged_energy = merged_energy[(merged_energy['datetime'] >= start_date) & (merged_energy['datetime'] <= end_date)]
# MERGE WITH WEATHER
weather['datetime'] = weather['datetime'].dt.floor('H')
combined = pd.merge(weather, merged_energy, on='datetime', how='inner')
combined.dropna(inplace=True)
combined = combined.sort_values('datetime').reset_index(drop=True)
#  FEATURE ENGINEERING
for lag in [1, 2, 3]:
    combined[f'TotalUsage_{lag}'] = combined['TotalUsage'].shift(lag)
combined.dropna(inplace=True)
combined = combined.sort_values('datetime').reset_index(drop=True)

print("Shape:", combined.shape)
y = combined['TotalUsage']
X = combined.drop(columns=['datetime', 'TotalUsage', 'serial'], errors='ignore')

for ii in range(1):
    for i in range(window_size, len(X)-horizon, step):
        X_train = X.iloc[i - window_size:i]
        y_train = y.iloc[i - window_size:i]
    
        X_test = X.iloc[i:i + horizon]
        y_test = y.iloc[i:i + horizon]
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        #OFDBAO
        best_C, best_gamma, best_epsilon, iterations, convCurve = ofdbao.OFDBAO(X_train_scaled,X_test_scaled,y_train,y_test, POP_SIZE, MAX_ITER, lower_bound, upper_bound, dim, i)
        svr = SVR(C=best_C, gamma=best_gamma, epsilon=best_epsilon)
        svr.fit(X_train_scaled, y_train)
        prediction = svr.predict(X_test_scaled)
        y_true.append(y_test.values[0])
        y_pred.append(prediction[0])
    
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.mean(np.minimum(np.abs((y_true - y_pred) / y_true), 1)) * 100
mape = mape(y_true, y_pred)
# Results
print("\n Test Results on Combined Dataset")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Assuming 'combined' has 'datetime' column and y_true, y_pred from rolling window
# Skip the first 'window_size' rows because they were used for training
dates = combined['datetime'].iloc[window_size:window_size + len(y_true)]
plt.figure(figsize=(7,4), dpi=350)
plt.plot(dates[:500], y_true[:500], color='k', label='Actual')
plt.plot(dates[:500], y_pred[:500], color='brown', label='Predicted', linestyle='--')
#plt.xlabel('Date')
plt.ylabel('Total Usage (kW)')
#plt.title(f'Rolling Window SVR: Actual vs Predicted\nRMSE: {np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2)):.2f}, R²: {np.corrcoef(y_true, y_pred)[0,1]**2:.3f}')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))  # e.g., 01-Nov
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # show ~10 ticks
plt.gcf().autofmt_xdate()  # rotate for readability
#plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Residual Plot
residuals = np.array(y_true) - np.array(y_pred)

plt.figure(figsize=(7, 4), dpi=350)  # increase resolution for publication
plt.scatter(y_pred, residuals, alpha=0.6, color='blue', edgecolor='k')  # colored points with edges
plt.axhline(0, color='r', linestyle='--', linewidth=2)  # zero line thicker for visibility
#plt.xlabel("Predicted Consumption")
plt.ylabel("Residuals")
#plt.title("Residuals vs Predicted Energy Consumption")  # optional title
plt.grid(True, linestyle='--', alpha=0.7)  # lighter dashed grid
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4), dpi=350)  # high-resolution figure
plt.scatter(np.array(y_true), np.array(y_pred), alpha=0.6, color='blue', edgecolor='k')  # scatter points with edges
# 45-degree line spanning the full range
min_val = min(np.min(y_true), np.min(y_pred))
max_val = max(np.max(y_true), np.max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
#plt.title("Actual vs Predicted Energy Consumption")  # optional title
plt.grid(True, linestyle='--', alpha=0.7)  # dashed grid for clarity
plt.tight_layout()
plt.show()