# backtest.py (OPTIMIZED VERSION - Fixes all warnings and performance issues)

import os
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm
import warnings

# Suppress the warnings you're seeing
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# We need to import the functions from our other scripts
from train import LSTMModel, Config, create_features
from utils import generate_probabilities, generate_confidence

def run_backtest_optimized(config):
    """
    OPTIMIZED version - Generates predictions much faster with batch processing
    """
    print("--- Starting OPTIMIZED Backtest Prediction Generation ---")
    print("Loading models, scalers, and data...")
    
    try:
        device = torch.device("cpu")
        model = LSTMModel(input_dim=config.FEATURE_COUNT, hidden_dim=100, output_dim=config.PREDICTION_STEPS, n_layers=2)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        feature_scaler = joblib.load(config.FEATURE_SCALER_PATH)
        label_scaler = joblib.load(config.LABEL_SCALER_PATH)
        
        main_df_path = os.path.join(config.DATA_DIR, f"{config.MAIN_SYMBOL_MQL5}.csv")
        
        # FIXED: Make this DataFrame timezone-aware to match the other one
        main_df = pd.read_csv(
            main_df_path, 
            index_col='Datetime', 
            parse_dates=True, 
            encoding='utf-8-sig', 
            sep='\t'
        )
        main_df.index = pd.to_datetime(main_df.index, utc=True)

    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a required file: {e}. Please run train.py first.")
        return

    # Create features for the entire dataset
    print("Creating features...")
    processed_data = create_features(main_df.copy(), config)
    
    feature_cols = [col for col in processed_data.columns if col.startswith('f')]
    X_full = processed_data[feature_cols].values
    X_full_scaled = feature_scaler.transform(X_full)

    print(f"Total samples to process: {len(X_full_scaled) - config.SEQ_LEN}")
    
    # OPTIMIZATION 1: Pre-allocate arrays and process in batches
    num_predictions = len(X_full_scaled) - config.SEQ_LEN
    batch_size = 256  # Process 256 predictions at once instead of 1
    
    predictions = []
    
    print("\n--- Generating predictions with BATCH PROCESSING ---")
    
    # OPTIMIZATION 2: Process in batches instead of one-by-one
    for batch_start in tqdm(range(0, num_predictions, batch_size), desc="Backtesting (Batched)"):
        batch_end = min(batch_start + batch_size, num_predictions)
        current_batch_size = batch_end - batch_start
        
        # Prepare batch data
        batch_sequences = []
        batch_timestamps = []
        batch_current_prices = []
        batch_atr_vals = []
        
        for i in range(batch_start, batch_end):
            feature_seq = X_full_scaled[i : i + config.SEQ_LEN]
            prediction_timestamp = processed_data.index[i + config.SEQ_LEN - 1]
            current_price = main_df.loc[prediction_timestamp]['Close']
            atr_val = processed_data.loc[prediction_timestamp]['f3_atr']
            
            batch_sequences.append(feature_seq)
            batch_timestamps.append(prediction_timestamp)
            batch_current_prices.append(current_price)
            batch_atr_vals.append(atr_val)
        
        # FIXED: Convert to numpy array first to avoid PyTorch warning
        batch_sequences_array = np.array(batch_sequences, dtype=np.float32)
        
        # FIXED: Use torch.from_numpy instead of torch.tensor for better performance
        features_tensor = torch.from_numpy(batch_sequences_array).to(device)
        
        # Get predictions for entire batch at once
        with torch.no_grad():
            batch_predictions_scaled = model(features_tensor).cpu().numpy()
        
        # Process each prediction in the batch
        for j in range(current_batch_size):
            prediction_scaled = batch_predictions_scaled[j:j+1]  # Keep 2D shape
            current_price = batch_current_prices[j]
            atr_val = batch_atr_vals[j]
            prediction_timestamp = batch_timestamps[j]
            
            # MAJOR FIX: Reconstruct price from predicted difference
            predicted_price_diffs = label_scaler.inverse_transform(prediction_scaled)[0]
            predicted_prices = current_price + predicted_price_diffs
            
            buy_prob, sell_prob = generate_probabilities(predicted_prices, current_price)
            hold_prob = 1.0 - buy_prob - sell_prob
            confidence = generate_confidence(predicted_prices, atr_val)
            
            result = {
                "timestamp": prediction_timestamp,
                "buy_prob": buy_prob,
                "sell_prob": sell_prob,
                "hold_prob": hold_prob,
                "confidence_score": confidence,
            }
            for step in range(config.PREDICTION_STEPS):
                result[f"predicted_price_h{step+1}"] = predicted_prices[step]
            predictions.append(result)

    if not predictions:
        print("❌ No predictions were generated. Check data processing steps.")
        return
    
    print(f"\n--- Processing {len(predictions)} predictions for output ---")
    results_df = pd.DataFrame(predictions)
    
    # Format timestamp for MQL5: YYYY.MM.DD HH:MM:SS
    results_df['timestamp'] = results_df['timestamp'].dt.strftime('%Y.%m.%d %H:%M:%S')
    
    # Rename columns to match expected format
    column_mapping = {
        'timestamp': 'timestamp',
        'buy_prob': 'buy_prob', 
        'sell_prob': 'sell_prob',
        'hold_prob': 'hold_prob',
        'confidence_score': 'confidence_score'
    }
    
    # Add prediction columns with simpler names
    for step in range(config.PREDICTION_STEPS):
        column_mapping[f'predicted_price_h{step+1}'] = f'predicted_price_{step}'
    
    results_df = results_df.rename(columns=column_mapping)
    
    # Reorder columns to match expected format
    expected_columns = ['timestamp', 'buy_prob', 'sell_prob', 'hold_prob', 'confidence_score'] + \
                      [f'predicted_price_{step}' for step in range(config.PREDICTION_STEPS)]
    
    results_df = results_df[expected_columns]
    
    output_path = "backtest_predictions.csv"
    print(f"\n--- Saving {len(results_df)} predictions to {output_path} ---")
    
    # Save with semicolon separator for MT5
    results_df.to_csv(output_path, sep=';', index=False, header=True, float_format='%.8f')
    
    # Try to copy to MT5 common files automatically
    try:
        common_files_path = os.path.join(os.getenv('APPDATA'), 'MetaQuotes', 'Terminal', 'Common', 'Files')
        if os.path.exists(common_files_path):
            mt5_output_path = os.path.join(common_files_path, 'backtest_predictions.csv')
            results_df.to_csv(mt5_output_path, sep=';', index=False, header=True, float_format='%.8f')
            print(f"✅ File automatically copied to MT5: {mt5_output_path}")
        else:
            print(f"📁 Manual copy needed to: {common_files_path}")
    except Exception as e:
        print(f"⚠️ Could not auto-copy to MT5 folder: {e}")
    
    print("\n✅ Backtest prediction file generated successfully!")
    print(f"📊 Generated {len(results_df)} predictions")
    print(f"💾 File saved as: {output_path}")
    
    # Show sample of generated data
    print("\n📋 Sample of generated predictions:")
    print(results_df.head())
    
    return results_df

def run_backtest(config):
    """
    Original function - kept for compatibility but calls optimized version
    """
    return run_backtest_optimized(config)

def quick_test_generation(num_samples=1000):
    """
    Generate a quick test file with sample data for immediate EA testing
    """
    print(f"🚀 GENERATING QUICK TEST FILE ({num_samples} samples)...")
    
    from datetime import datetime, timedelta
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=num_samples)
    timestamps = [start_time + timedelta(hours=i) for i in range(num_samples)]
    
    # Generate sample predictions
    data = []
    base_price = 1.0800
    
    for i, timestamp in enumerate(timestamps):
        # Simple price progression with some variation
        price_variation = np.sin(i * 0.1) * 0.002  # Creates realistic price movement
        current_price = base_price + (i * 0.00001) + price_variation
        
        # Generate realistic predictions
        predictions = []
        for step in range(5):  # 5 prediction steps
            step_change = np.random.normal(0, 0.0001 * (step + 1))
            trend_component = 0.0002 if np.random.random() > 0.5 else -0.0002
            predicted_price = current_price + (step_change + trend_component * (step + 1) * 0.1)
            predictions.append(predicted_price)
        
        # Generate realistic probabilities
        buy_prob = max(0.1, min(0.8, np.random.normal(0.4, 0.15)))
        sell_prob = max(0.1, min(0.8, np.random.normal(0.3, 0.15))) 
        total_prob = buy_prob + sell_prob
        if total_prob > 0.9:
            buy_prob *= 0.9 / total_prob
            sell_prob *= 0.9 / total_prob
        hold_prob = 1.0 - buy_prob - sell_prob
        
        confidence = max(0.3, min(0.95, np.random.normal(0.65, 0.15)))
        
        row = [
            timestamp.strftime('%Y.%m.%d %H:%M:%S'),
            round(buy_prob, 6),
            round(sell_prob, 6),
            round(hold_prob, 6), 
            round(confidence, 6)
        ] + [round(p, 5) for p in predictions]
        
        data.append(row)
    
    # Create DataFrame
    columns = ['timestamp', 'buy_prob', 'sell_prob', 'hold_prob', 'confidence_score'] + \
             [f'predicted_price_{i}' for i in range(5)]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save files
    output_file = 'backtest_predictions.csv'
    df.to_csv(output_file, sep=';', index=False)
    print(f"✅ Test file saved: {output_file}")
    
    # Try to save to MT5 folder
    try:
        common_files_path = os.path.join(os.getenv('APPDATA'), 'MetaQuotes', 'Terminal', 'Common', 'Files')
        if os.path.exists(common_files_path):
            mt5_path = os.path.join(common_files_path, 'backtest_predictions.csv')
            df.to_csv(mt5_path, sep=';', index=False)
            print(f"✅ File copied to MT5: {mt5_path}")
    except:
        pass
    
    print(f"📊 Generated {len(df)} sample predictions")
    print("🚀 You can now test your MT5 EA immediately!")
    
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick test generation
        quick_test_generation()
    else:
        # Full backtest
        cfg = Config()
        run_backtest_optimized(cfg)