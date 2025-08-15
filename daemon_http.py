import os
import json
import time
import sys
import numpy as np
import torch
import joblib
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import logging
from urllib.parse import urlparse, parse_qs
from train import LSTMModel, Config
from utils import generate_probabilities, generate_confidence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- GLOBAL VARIABLES & CONFIG ---
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
feature_scaler = None
label_scaler = None

# Server configuration
SERVER_HOST = "127.0.0.1"  # localhost only for security
SERVER_PORT = 8888
MAX_REQUEST_SIZE = 1024 * 1024  # 1MB max request size

# Statistics tracking
request_count = 0
successful_predictions = 0
failed_predictions = 0
start_time = time.time()

class PredictionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to use our logger instead of stderr"""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def do_GET(self):
        """Handle GET requests for health check and stats"""
        global request_count, successful_predictions, failed_predictions, start_time
        
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            health_data = {
                "status": "healthy",
                "model_loaded": model is not None,
                "scalers_loaded": feature_scaler is not None and label_scaler is not None,
                "device": str(device),
                "uptime_seconds": int(time.time() - start_time),
                "seq_len": cfg.SEQ_LEN,
                "feature_count": cfg.FEATURE_COUNT,
                "prediction_steps": cfg.PREDICTION_STEPS
            }
            
            self.wfile.write(json.dumps(health_data, indent=2).encode())
            
        elif parsed_path.path == '/stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            stats_data = {
                "total_requests": request_count,
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "success_rate": (successful_predictions / max(request_count, 1)) * 100,
                "uptime_seconds": int(time.time() - start_time),
                "requests_per_minute": (request_count / max((time.time() - start_time) / 60, 1)),
                "average_processing_time": 0.05  # placeholder
            }
            
            self.wfile.write(json.dumps(stats_data, indent=2).encode())
            
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_POST(self):
        """Handle POST requests for predictions"""
        global request_count, successful_predictions, failed_predictions
        
        # Route to correct endpoint
        parsed_path = urlparse(self.path)
        
        if parsed_path.path not in ['/predict', '/']:
            self.send_error(404, "Endpoint not found")
            return
        
        request_count += 1
        
        try:
            # Check content length
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > MAX_REQUEST_SIZE:
                self.send_error(413, "Request too large")
                return
            
            if content_length == 0:
                self.send_error(400, "Empty request body")
                return
            
            # Read and parse request data
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                self.send_error(400, f"Invalid JSON: {str(e)}")
                return
            
            # Process the prediction request
            start_time_processing = time.time()
            response_data = process_request(request_data)
            processing_time = (time.time() - start_time_processing) * 1000  # Convert to ms
            
            response_data["processing_time_ms"] = round(processing_time, 2)
            
            # Send response
            if response_data.get("status") == "success":
                successful_predictions += 1
                self.send_response(200)
            else:
                failed_predictions += 1
                self.send_response(500)
            
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_json = json.dumps(response_data, indent=2)
            self.wfile.write(response_json.encode())
            
            logger.info(f"Request processed in {processing_time:.2f}ms. Status: {response_data.get('status')}")
            
        except Exception as e:
            failed_predictions += 1
            logger.error(f"Error processing request: {e}")
            
            try:
                error_response = {
                    "status": "error",
                    "message": f"Internal server error: {str(e)}",
                    "error_type": type(e).__name__
                }
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
            except:
                # Connection might be broken
                pass
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def verify_paths_and_permissions():
    """Verify model files exist and are accessible"""
    logger.info("Verifying model files and permissions...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, cfg.MODEL_PATH)
    feature_scaler_path = os.path.join(script_dir, cfg.FEATURE_SCALER_PATH)
    label_scaler_path = os.path.join(script_dir, cfg.LABEL_SCALER_PATH)
    
    missing_files = []
    for path, name in [(model_path, "Model"), (feature_scaler_path, "Feature Scaler"), (label_scaler_path, "Label Scaler")]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        logger.error("Missing required files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        logger.error("Please run 'python train.py' to generate model files.")
        return False
    
    logger.info("All required files found.")
    return True

def load_models_and_scalers():
    """Load the ML models and scalers"""
    global model, feature_scaler, label_scaler
    
    logger.info("Loading models and scalers...")
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, cfg.MODEL_PATH)
        feature_scaler_path = os.path.join(script_dir, cfg.FEATURE_SCALER_PATH)
        label_scaler_path = os.path.join(script_dir, cfg.LABEL_SCALER_PATH)

        # Load model
        model = LSTMModel(
            input_dim=cfg.FEATURE_COUNT, 
            hidden_dim=100, 
            output_dim=cfg.PREDICTION_STEPS, 
            n_layers=2
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Model '{cfg.MODEL_PATH}' loaded successfully on {device}.")
        
        # Load scalers
        feature_scaler = joblib.load(feature_scaler_path)
        logger.info(f"Feature scaler '{cfg.FEATURE_SCALER_PATH}' loaded successfully.")

        label_scaler = joblib.load(label_scaler_path)
        logger.info(f"Label scaler '{cfg.LABEL_SCALER_PATH}' loaded successfully.")
        
        logger.info("All models and scalers loaded successfully.")
        logger.info(f"Model configuration: {cfg.FEATURE_COUNT} features, {cfg.SEQ_LEN} sequence length, {cfg.PREDICTION_STEPS} prediction steps")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def process_request(request_data):
    """Process a prediction request"""
    try:
        # Validate request data
        required_fields = ['features', 'current_price', 'atr']
        for field in required_fields:
            if field not in request_data:
                return {
                    "status": "error", 
                    "message": f"Missing required field: {field}",
                    "required_fields": required_fields
                }
        
        features_flat = request_data.get('features', [])
        current_price = request_data.get('current_price', 0)
        atr_val = request_data.get('atr', 0)
        
        # Validate features array
        expected_size = cfg.SEQ_LEN * cfg.FEATURE_COUNT
        if not features_flat or len(features_flat) != expected_size:
            return {
                "status": "error", 
                "message": f"Invalid features array. Expected {expected_size} elements, got {len(features_flat)}",
                "expected_size": expected_size,
                "received_size": len(features_flat),
                "seq_len": cfg.SEQ_LEN,
                "feature_count": cfg.FEATURE_COUNT
            }

        # Validate numeric inputs
        try:
            current_price = float(current_price)
            atr_val = float(atr_val)
            features_flat = [float(f) for f in features_flat]
        except (ValueError, TypeError) as e:
            return {
                "status": "error", 
                "message": f"Invalid numeric data: {str(e)}"
            }

        # Validate reasonable ranges
        if current_price <= 0:
            return {
                "status": "error",
                "message": "Current price must be positive"
            }
        
        if atr_val < 0:
            return {
                "status": "error",
                "message": "ATR value must be non-negative"
            }

        # Reshape and scale features
        features_np = np.array(features_flat).reshape(cfg.SEQ_LEN, cfg.FEATURE_COUNT)
        features_scaled = feature_scaler.transform(features_np)
        features_tensor = torch.tensor([features_scaled], dtype=torch.float32).to(device)
        
        # Generate prediction
        with torch.no_grad():
            prediction_scaled = model(features_tensor).cpu().numpy()
        
        # --- MAJOR FIX: Reconstruct price from predicted difference ---
        # The model now predicts the difference from the current price.
        # We inverse_transform the scaled difference, then add it to the current price.
        predicted_price_diffs = label_scaler.inverse_transform(prediction_scaled)[0]
        predicted_prices = current_price + predicted_price_diffs
        
        # Generate additional metrics
        buy_prob, sell_prob = generate_probabilities(predicted_prices, current_price)
        confidence = generate_confidence(predicted_prices, atr_val)
        
        # Validate predictions
        if np.any(np.isnan(predicted_prices)) or np.any(np.isinf(predicted_prices)):
            return {
                "status": "error",
                "message": "Model generated invalid predictions (NaN or Inf values)"
            }

        return {
            "status": "success",
            "predicted_prices": predicted_prices.tolist(),
            "confidence_score": float(confidence),
            "buy_probability": float(buy_prob),
            "sell_probability": float(sell_prob),
            "request_id": request_data.get('request_id', 'unknown'),
            "current_price": current_price,
            "atr_value": atr_val,
            "model_info": {
                "device": str(device),
                "seq_len": cfg.SEQ_LEN,
                "feature_count": cfg.FEATURE_COUNT,
                "prediction_steps": cfg.PREDICTION_STEPS
            }
        }
        
    except Exception as e:
        logger.error(f"Error in process_request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error", 
            "message": f"Prediction processing error: {str(e)}",
            "error_type": type(e).__name__
        }

def start_server():
    """Start the HTTP server"""
    try:
        server = HTTPServer((SERVER_HOST, SERVER_PORT), PredictionHandler)
        logger.info("=" * 60)
        logger.info("LSTM Trading Daemon - HTTP Server Started")
        logger.info("=" * 60)
        logger.info(f"Server URL: http://{SERVER_HOST}:{SERVER_PORT}")
        logger.info("Available endpoints:")
        logger.info(f"  POST http://{SERVER_HOST}:{SERVER_PORT}/predict - Make predictions")
        logger.info(f"  GET  http://{SERVER_HOST}:{SERVER_PORT}/health  - Health check")
        logger.info(f"  GET  http://{SERVER_HOST}:{SERVER_PORT}/stats   - Server statistics")
        logger.info("=" * 60)
        logger.info("Server is ready to accept requests...")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 60)
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("\nServer shutdown requested by user")
        server.shutdown()
        logger.info("Server stopped successfully")
    except OSError as e:
        if e.errno == 10048:  # Address already in use
            logger.error(f"Port {SERVER_PORT} is already in use. Please check if another daemon is running.")
            logger.error("To kill existing process: netstat -ano | findstr :8888")
        else:
            logger.error(f"Server error: {e}")
    except Exception as e:
        logger.error(f"Unexpected server error: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_server():
    """Test the server with a sample request"""
    import requests
    import time
    
    logger.info("Testing server with sample request...")
    
    # Wait a moment for server to start
    time.sleep(2)
    
    try:
        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
        
        # Test health endpoint
        logger.info("Testing health endpoint...")
        health_response = requests.get(f"{base_url}/health", timeout=5)
        logger.info(f"Health check status: {health_response.status_code}")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            logger.info(f"Health response: {json.dumps(health_data, indent=2)}")
        
        # Test prediction endpoint
        logger.info("Testing prediction endpoint...")
        test_features = [0.001 * (i % 100) for i in range(cfg.SEQ_LEN * cfg.FEATURE_COUNT)]
        test_request = {
            "request_id": "test_001",
            "current_price": 1.1234,
            "atr": 0.0012,
            "features": test_features
        }
        
        pred_response = requests.post(
            f"{base_url}/predict", 
            json=test_request,
            timeout=10
        )
        
        logger.info(f"Prediction status: {pred_response.status_code}")
        
        if pred_response.status_code == 200:
            result = pred_response.json()
            logger.info("✅ Test prediction successful!")
            logger.info(f"Confidence: {result.get('confidence_score', 'N/A')}")
            logger.info(f"Processing time: {result.get('processing_time_ms', 'N/A')}ms")
            logger.info(f"Predicted prices: {result.get('predicted_prices', [])[:3]}...")  # Show first 3
        else:
            logger.error(f"❌ Test prediction failed: {pred_response.status_code}")
            logger.error(pred_response.text)
        
        # Test stats endpoint
        logger.info("Testing stats endpoint...")
        stats_response = requests.get(f"{base_url}/stats", timeout=5)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            logger.info(f"Server stats: {json.dumps(stats_data, indent=2)}")
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to server. Make sure it's running.")
    except requests.exceptions.Timeout:
        logger.error("❌ Server request timed out.")
    except Exception as e:
        logger.error(f"❌ Server test failed: {e}")

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("LSTM Trading Daemon - HTTP Server Version")
    logger.info("=" * 60)
    
    # Verify files exist
    if not verify_paths_and_permissions():
        logger.error("Pre-flight checks failed. Exiting.")
        sys.exit(1)
    
    # Load models
    if not load_models_and_scalers():
        logger.error("Failed to load models. Exiting.")
        sys.exit(1)
    
    # Start test in separate thread if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("Test mode enabled - will run self-test after startup")
        test_thread = threading.Thread(target=test_server)
        test_thread.daemon = True
        test_thread.start()
    
    # Start the server
    try:
        start_server()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()