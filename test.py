# test_communication.py - Standalone test to verify daemon communication

import os
import json
import time
import random
from train import Config

def find_mt5_files_path():
    """Find MT5 files directory"""
    possible_paths = [
        r"C:\Users\jason\AppData\Roaming\MetaQuotes\Terminal",
        os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal")
    ]
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            for folder in os.listdir(base_path):
                if len(folder) == 32:  # Terminal ID folders are 32 chars
                    files_path = os.path.join(base_path, folder, "MQL5", "Files")
                    if os.path.exists(files_path):
                        return files_path
    return None

def create_test_request(files_dir, test_id="test001"):
    """Create a properly formatted test request"""
    cfg = Config()
    
    # Generate random but realistic features
    features = []
    for seq in range(cfg.SEQ_LEN):  # 20 sequences
        for feat in range(cfg.FEATURE_COUNT):  # 15 features each
            if feat == 0:  # close_return
                features.append(random.uniform(-0.01, 0.01))
            elif feat == 1:  # tick_volume  
                features.append(random.uniform(100, 1000))
            elif feat == 2:  # atr
                features.append(random.uniform(0.0001, 0.0020))
            elif feat == 3:  # macd
                features.append(random.uniform(-0.0010, 0.0010))
            elif feat == 4:  # rsi
                features.append(random.uniform(20, 80))
            elif feat == 5:  # stoch_k
                features.append(random.uniform(0, 100))
            elif feat == 6:  # cci
                features.append(random.uniform(-200, 200))
            elif feat == 7:  # hour
                features.append(random.randint(0, 23))
            elif feat == 8:  # day_of_week
                features.append(random.randint(0, 6))
            else:  # other features
                features.append(random.uniform(-1, 1))
    
    request_data = {
        "request_id": test_id,
        "features": features,
        "current_price": 1.10000 + random.uniform(-0.001, 0.001),
        "atr": random.uniform(0.0008, 0.0012)
    }
    
    filename = f"request_{test_id}.json"
    filepath = os.path.join(files_dir, filename)
    
    print(f"Creating test request: {filename}")
    print(f"Features count: {len(features)}")
    print(f"Current price: {request_data['current_price']}")
    print(f"ATR: {request_data['atr']}")
    
    with open(filepath, 'w') as f:
        json.dump(request_data, f, indent=2)
    
    return filename, filepath

def wait_for_response(files_dir, test_id, timeout=30):
    """Wait for daemon response"""
    response_filename = f"response_{test_id}.json"
    response_filepath = os.path.join(files_dir, response_filename)
    
    print(f"Waiting for response: {response_filename}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(response_filepath):
            try:
                with open(response_filepath, 'r') as f:
                    response_data = json.load(f)
                
                print("✅ Response received!")
                print(f"Status: {response_data.get('status')}")
                
                if response_data.get('status') == 'success':
                    print(f"Buy probability: {response_data.get('buy_probability', 'N/A')}")
                    print(f"Sell probability: {response_data.get('sell_probability', 'N/A')}")
                    print(f"Confidence: {response_data.get('confidence_score', 'N/A')}")
                    predicted_prices = response_data.get('predicted_prices', [])
                    if predicted_prices:
                        print(f"Predicted prices (5 steps): {predicted_prices}")
                else:
                    print(f"Error: {response_data.get('message', 'Unknown error')}")
                
                # Clean up response file
                try:
                    os.remove(response_filepath)
                    print("Response file cleaned up")
                except:
                    pass
                
                return True
                
            except Exception as e:
                print(f"Error reading response: {e}")
                return False
        
        time.sleep(0.5)
    
    print(f"❌ Timeout waiting for response after {timeout} seconds")
    return False

def test_daemon_communication():
    """Main test function"""
    print("=== Daemon Communication Test ===\n")
    
    # Find MT5 files directory
    files_dir = find_mt5_files_path()
    if not files_dir:
        print("❌ Could not find MT5 files directory!")
        print("Make sure MetaTrader 5 is installed and has been opened at least once.")
        return False
    
    print(f"✅ Found MT5 files directory: {files_dir}")
    
    # Check if daemon is likely running by looking for any response files
    existing_responses = [f for f in os.listdir(files_dir) if f.startswith('response_')]
    if existing_responses:
        print(f"Found {len(existing_responses)} existing response files (daemon might be running)")
    
    # Test 1: Basic communication test
    print("\n--- Test 1: Basic Communication ---")
    test_id = f"basic_{int(time.time())}"
    
    try:
        request_file, request_path = create_test_request(files_dir, test_id)
        
        # Wait a moment for daemon to pick up the file
        time.sleep(1)
        
        # Check if request file was consumed (daemon should delete it)
        if os.path.exists(request_path):
            print("⚠️  Request file still exists - daemon might not be running")
        else:
            print("✅ Request file was consumed by daemon")
        
        # Wait for response
        success = wait_for_response(files_dir, test_id)
        
        if success:
            print("✅ Test 1 PASSED: Basic communication working")
        else:
            print("❌ Test 1 FAILED: No response received")
            return False
            
    except Exception as e:
        print(f"❌ Test 1 FAILED with exception: {e}")
        return False
    
    # Test 2: Rapid fire test (3 requests quickly)
    print("\n--- Test 2: Rapid Fire (3 requests) ---")
    
    test_ids = []
    for i in range(3):
        test_id = f"rapid_{int(time.time())}_{i}"
        test_ids.append(test_id)
        create_test_request(files_dir, test_id)
        time.sleep(0.1)  # Small delay between requests
    
    # Wait for all responses
    all_success = True
    for test_id in test_ids:
        if not wait_for_response(files_dir, test_id, timeout=15):
            all_success = False
    
    if all_success:
        print("✅ Test 2 PASSED: Rapid fire requests handled correctly")
    else:
        print("❌ Test 2 FAILED: Some rapid fire requests failed")
    
    # Test 3: Invalid request test
    print("\n--- Test 3: Invalid Request Handling ---")
    
    invalid_request = {
        "request_id": "invalid_test",
        "features": [1, 2, 3],  # Wrong number of features
        "current_price": 1.1000,
        "atr": 0.0010
    }
    
    invalid_filename = "request_invalid_test.json"
    invalid_filepath = os.path.join(files_dir, invalid_filename)
    
    with open(invalid_filepath, 'w') as f:
        json.dump(invalid_request, f)
    
    print("Created invalid request (wrong feature count)")
    
    success = wait_for_response(files_dir, "invalid_test", timeout=10)
    if success:
        print("✅ Test 3 PASSED: Invalid request handled gracefully")
    else:
        print("❌ Test 3 FAILED: Invalid request not handled")
    
    print("\n=== Communication Test Complete ===")
    return True

def check_daemon_requirements():
    """Check if all requirements are met for daemon"""
    print("=== Checking Daemon Requirements ===\n")
    
    cfg = Config()
    issues = []
    
    # Check model files
    model_files = [
        cfg.MODEL_PATH,
        cfg.FEATURE_SCALER_PATH, 
        cfg.LABEL_SCALER_PATH
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            issues.append(f"Missing {file_path}")
    
    # Check if daemon script exists
    daemon_scripts = ['daemon.py', 'daemon_debug.py']
    daemon_found = False
    for script in daemon_scripts:
        if os.path.exists(script):
            print(f"✅ Found daemon script: {script}")
            daemon_found = True
            break
    
    if not daemon_found:
        issues.append("No daemon script found")
    
    # Check Python dependencies
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        issues.append("PyTorch not installed")
    
    try:
        import joblib
        print("✅ joblib available")
    except ImportError:
        issues.append("joblib not installed")
    
    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before running the daemon.")
        return False
    else:
        print("\n✅ All requirements met!")
        return True

if __name__ == "__main__":
    print("=== Daemon Communication Diagnostic Tool ===\n")
    
    # First check requirements
    if not check_daemon_requirements():
        print("\n❌ Requirements not met. Please run 'python train.py' first if models are missing.")
        exit(1)
    
    print("\n" + "="*50)
    
    # Then test communication
    if test_daemon_communication():
        print("\n✅ Communication tests completed successfully!")
        print("\nIf tests passed but your EA still can't communicate:")
        print("1. Check that your EA creates JSON files in the exact same format")
        print("2. Verify your EA is writing to the correct MT5 Files directory")
        print("3. Make sure your EA waits for response files before reading them")
    else:
        print("\n❌ Communication tests failed!")
        print("\nTroubleshooting steps:")
        print("1. Make sure the daemon is running: 'python daemon_debug.py'")
        print("2. Check if you have write permissions to the MT5 Files directory")
        print("3. Verify all model files exist (run 'python train.py' if needed)")
    
    input("\nPress Enter to exit...")