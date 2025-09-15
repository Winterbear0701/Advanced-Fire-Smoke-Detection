"""
Advanced Fire & Smoke Detection System - Demo Script
This script demonstrates the new API endpoints and features
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://127.0.0.1:8080"
TEST_IMAGE_PATH = "data/1.jpg"  # Update with your test image path

def test_api_endpoints():
    """Test all the new API endpoints"""
    
    print("🔥 Advanced Fire & Smoke Detection System - API Demo")
    print("=" * 60)
    
    # Test 1: Get available models
    print("\n1. Testing /api/models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Available models: {len(models['models'])}")
            for model in models['models']:
                print(f"   - {model['name']}: {model['description']}")
        else:
            print(f"❌ Failed to get models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Get detection history
    print("\n2. Testing /api/history endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/history?limit=5")
        if response.status_code == 200:
            history = response.json()
            print(f"✅ History records: {len(history['history'])}")
            for record in history['history'][:3]:
                print(f"   - {record.get('timestamp', 'N/A')}: {record.get('detection_count', 0)} detections")
        else:
            print(f"❌ Failed to get history: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Get detection statistics
    print("\n3. Testing /api/stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Statistics retrieved successfully")
            if 'stats' in stats and stats['stats']:
                for key, value in stats['stats'].items():
                    if isinstance(value, dict):
                        print(f"   - {key}: {value}")
                    else:
                        print(f"   - {key}: {value}")
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Advanced detection with different models
    if Path(TEST_IMAGE_PATH).exists():
        print(f"\n4. Testing advanced detection with {TEST_IMAGE_PATH}...")
        
        models_to_test = ['yolov8n', 'yolov8s']
        confidences = [0.3, 0.5, 0.7]
        
        for model in models_to_test:
            for confidence in confidences:
                print(f"\n   Testing {model} with confidence {confidence}...")
                try:
                    with open(TEST_IMAGE_PATH, 'rb') as f:
                        files = {'file': f}
                        data = {
                            'model_type': model,
                            'confidence': confidence,
                            'save_results': 'true'
                        }
                        
                        start_time = time.time()
                        response = requests.post(f"{BASE_URL}/detect", files=files, data=data)
                        processing_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get('success'):
                                print(f"   ✅ Detection completed in {processing_time:.2f}s")
                                print(f"      - Detections: {result.get('detection_count', 0)}")
                                print(f"      - Max confidence: {result.get('max_confidence', 'N/A')}")
                                print(f"      - Risk level: {result.get('risk_level', 'N/A')}")
                                print(f"      - Fire count: {result.get('fire_count', 0)}")
                                print(f"      - Smoke count: {result.get('smoke_count', 0)}")
                            else:
                                print(f"   ❌ Detection failed: {result.get('message', 'Unknown error')}")
                        else:
                            print(f"   ❌ HTTP Error: {response.status_code}")
                            
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                
                time.sleep(1)  # Brief pause between tests
    else:
        print(f"\n4. ⚠️ Test image not found at {TEST_IMAGE_PATH}")
        print("   Please place a test image in the data folder to test detection")
    
    print("\n" + "=" * 60)
    print("🎉 API Demo Complete!")
    print("\n💡 Feature Highlights:")
    print("   - Multiple AI models (YOLOv8n, YOLOv8s)")
    print("   - Adjustable confidence thresholds")
    print("   - Detailed detection analytics")
    print("   - Risk level assessment")
    print("   - Detection history tracking")
    print("   - RESTful API endpoints")
    print("   - Real-time processing capabilities")
    
    print("\n🌐 Access the web interface at: http://127.0.0.1:8080")
    print("📊 Try uploading images/videos through the modern web UI!")

def test_batch_processing():
    """Test batch processing capabilities"""
    print("\n5. Testing batch processing simulation...")
    
    # Simulate multiple concurrent requests
    if Path(TEST_IMAGE_PATH).exists():
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def process_image(model_type, confidence, result_queue):
            try:
                with open(TEST_IMAGE_PATH, 'rb') as f:
                    files = {'file': f}
                    data = {
                        'model_type': model_type,
                        'confidence': confidence,
                        'save_results': 'false'
                    }
                    
                    response = requests.post(f"{BASE_URL}/detect", files=files, data=data)
                    result_queue.put((model_type, confidence, response.json() if response.status_code == 200 else None))
            except Exception as e:
                result_queue.put((model_type, confidence, f"Error: {e}"))
        
        # Start multiple threads
        threads = []
        test_combinations = [
            ('yolov8n', 0.3),
            ('yolov8n', 0.5),
            ('yolov8s', 0.3),
            ('yolov8s', 0.5)
        ]
        
        start_time = time.time()
        
        for model, conf in test_combinations:
            thread = threading.Thread(target=process_image, args=(model, conf, results_queue))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        print(f"   ✅ Batch processing completed in {total_time:.2f}s")
        print(f"   📊 Processed {len(test_combinations)} requests concurrently")
        
        # Collect results
        while not results_queue.empty():
            model, conf, result = results_queue.get()
            if isinstance(result, dict) and result.get('success'):
                print(f"   - {model} (conf={conf}): {result.get('detection_count', 0)} detections")
            else:
                print(f"   - {model} (conf={conf}): Failed")

if __name__ == "__main__":
    # Wait a moment for the server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    # Run the demo
    test_api_endpoints()
    test_batch_processing()
    
    print("\n🚀 The system is now running with all advanced features!")
    print("🔗 Open http://127.0.0.1:8080 in your browser to explore the modern UI")
