import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- CONFIGURATION ---
class Config:
    SYMBOLS = ["EURUSD", "EURJPY", "USDJPY", "GBPUSD", "EURGBP", "USDCAD", "USDCHF"]
    MAIN_SYMBOL_MQL5 = "EURUSD"
    SEQ_LEN = 20
    PREDICTION_STEPS = 5
    FEATURE_COUNT = 15
    VALIDATION_SPLIT = 0.15
    EPOCHS = 30
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    DATA_DIR = "data"
    MODELS_DIR = "models"
    MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model_regression.pth")
    FEATURE_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_features.pkl")
    LABEL_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_labels.pkl")

# --- 1. DATA CHECKING ---
def check_local_data(config):
    print("--- Verifying local data files ---")
    all_files_exist = True
    for symbol in config.SYMBOLS:
        filepath = os.path.join(config.DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(filepath):
            print(f"❌ ERROR: Data file not found: {filepath}")
            print("Please run the 'ExportHistory.mq5' script in MetaTrader 5 first.")
            all_files_exist = False
    if all_files_exist:
        print("✅ All required data files found.")
    return all_files_exist

# --- 2. FEATURE ENGINEERING ---
def create_features(df, config):
    print("\n--- Engineering Features ---")
    aux_dfs = {}
    for sym_name in config.SYMBOLS:
        if sym_name != config.MAIN_SYMBOL_MQL5:
            path = os.path.join(config.DATA_DIR, f"{sym_name}.csv")
            if os.path.exists(path):
                aux_dfs[sym_name] = pd.read_csv(path, index_col='Datetime', parse_dates=True, encoding='utf-8-sig', sep='\t')
            else:
                print(f"Warning: Missing auxiliary data for {sym_name}")
                return pd.DataFrame()

    df.index = pd.to_datetime(df.index, utc=True)
    for key in aux_dfs:
        aux_dfs[key].index = pd.to_datetime(aux_dfs[key].index, utc=True)
        df = pd.merge(df, aux_dfs[key][['Close']].rename(columns={'Close': f'close_{key}'}), left_index=True, right_index=True, how='left')
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    features = pd.DataFrame(index=df.index)
    
    # Calculate indicators
    df.ta.atr(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(append=True)
    df.ta.stoch(append=True)
    df.ta.cci(append=True)
    df.ta.bbands(length=20, append=True)
    
    # --- ROBUST FIX: Find column names dynamically ---
    atr_col = next((col for col in df.columns if col.startswith('ATRr')), None)
    macd_col = next((col for col in df.columns if col.startswith('MACD_')), None)
    rsi_col = next((col for col in df.columns if col.startswith('RSI_')), None)
    stoch_k_col = next((col for col in df.columns if col.startswith('STOCHk_')), None)
    cci_col = next((col for col in df.columns if col.startswith('CCI_')), None)
    bbu_col = next((col for col in df.columns if col.startswith('BBU_')), None)
    bbl_col = next((col for col in df.columns if col.startswith('BBL_')), None)

    # Check if all columns were found
    if not all([atr_col, macd_col, rsi_col, stoch_k_col, cci_col, bbu_col, bbl_col]):
        print("❌ ERROR: Could not find one or more indicator columns after calculation.")
        print(f"Found columns: {df.columns.tolist()}")
        return pd.DataFrame()

    features['f1_close_return'] = df['Close'].pct_change()
    features['f2_tick_volume'] = df['Volume']
    features['f3_atr'] = df[atr_col]
    features['f4_macd'] = df[macd_col]
    features['f5_rsi'] = df[rsi_col]
    features['f6_stoch_k'] = df[stoch_k_col]
    features['f7_cci'] = df[cci_col]
    features['f8_hour'] = df.index.hour
    features['f9_day_of_week'] = df.index.dayofweek
    usd_jpy_ret = df['close_USDJPY'].pct_change()
    usd_cad_ret = df['close_USDCAD'].pct_change()
    usd_chf_ret = df['close_USDCHF'].pct_change()
    gbp_usd_ret = df['close_GBPUSD'].pct_change()
    features['f10_usd_basket_diff'] = (usd_jpy_ret + usd_cad_ret + usd_chf_ret) - (features['f1_close_return'] + gbp_usd_ret)
    eur_jpy_ret = df['close_EURJPY'].pct_change()
    eur_gbp_ret = df['close_EURGBP'].pct_change()
    features['f11_eur_basket'] = features['f1_close_return'] + eur_jpy_ret + eur_gbp_ret
    features['f12_jpy_basket'] = -(eur_jpy_ret + usd_jpy_ret)
    features['f13_bb_width'] = (df[bbu_col] - df[bbl_col]) / (df['Close'] + 1e-10)
    features['f14_volume_roc'] = df['Volume'].diff(periods=5)
    body = (df['Close'] - df['Open']).abs()
    range_val = df['High'] - df['Low']
    bar_type = pd.Series(np.zeros(len(df)), index=df.index)
    bar_type[range_val > 0] = (body / range_val) < 0.1
    is_bullish_engulfing = (df['Close'] > df['Open']) & (df['Open'] < df['Open'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    is_bearish_engulfing = (df['Close'] < df['Open']) & (df['Open'] > df['Open'].shift(1)) & (df['Close'] < df['Close'].shift(1))
    bar_type[is_bullish_engulfing] = 2.0
    bar_type[is_bearish_engulfing] = -2.0
    gap_change = (df['Open'] - df['Close'].shift(1)) / (df[atr_col] + 1e-10)
    bar_type += gap_change
    features['f15_bar_type'] = bar_type
    
    labels = pd.DataFrame(index=df.index)
    # --- MAJOR FIX: Predict price DIFFERENCE, not absolute price ---
    # This fixes the scaling issue where predictions are out of the current market range.
    for i in range(1, config.PREDICTION_STEPS + 1):
        labels[f'label_h{i}'] = df['Close'].shift(-i) - df['Close']
    
    full_data = pd.concat([features, labels], axis=1)
    full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_data.dropna(inplace=True)
    print(f"Feature engineering complete. Shape: {full_data.shape}")
    return full_data

# --- 3. PYTORCH DATASET & MODEL ---
class ForexDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
    def __len__(self):
        return len(self.features) - self.seq_len
    def __getitem__(self, idx):
        feature_seq = self.features[idx:idx + self.seq_len]
        label = self.labels[idx + self.seq_len - 1]
        return torch.tensor(feature_seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out

# --- 4. TRAINING SCRIPT ---
def run_training(config):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    main_df_path = os.path.join(config.DATA_DIR, f"{config.MAIN_SYMBOL_MQL5}.csv")
    main_df = pd.read_csv(main_df_path, index_col='Datetime', parse_dates=True, encoding='utf-8-sig', sep='\t')
    processed_data = create_features(main_df, config)
    if processed_data.empty:
        print("Stopping training due to data processing errors.")
        return

    feature_cols = [col for col in processed_data.columns if col.startswith('f')]
    label_cols = [col for col in processed_data.columns if col.startswith('label')]
    X = processed_data[feature_cols].values
    y = processed_data[label_cols].values

    print("\n--- Scaling Data ---")
    feature_scaler = StandardScaler()
    label_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = label_scaler.fit_transform(y)
    joblib.dump(feature_scaler, config.FEATURE_SCALER_PATH)
    joblib.dump(label_scaler, config.LABEL_SCALER_PATH)
    print(f"Scalers saved to {config.MODELS_DIR}")

    split_idx = int(len(X_scaled) * (1 - config.VALIDATION_SPLIT))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
    train_dataset = ForexDataset(X_train, y_train, config.SEQ_LEN)
    val_dataset = ForexDataset(X_val, y_val, config.SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = LSTMModel(input_dim=config.FEATURE_COUNT, hidden_dim=100, output_dim=config.PREDICTION_STEPS, n_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("\n--- Starting Model Training ---")
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]")
        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({"Loss": loss.item()})
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Validation]")
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix({"Loss": loss.item()})
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{config.EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"\n--- Training Complete. Model saved to {config.MODEL_PATH} ---")

if __name__ == "__main__":
    cfg = Config()
    if check_local_data(cfg):
        run_training(cfg)