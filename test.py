import requests
import json

# API base URL
BASE_URL = "http://localhost:6000"

def test_single_prediction():
    """Test single text prediction"""
    url = f"{BASE_URL}/predict"
    
    test_cases = [
        "All cryptocurrency is just a Ponzi scheme with no real value",
        "Cryptocurrency value depends on various factors including technology, adoption, and market conditions",
        "Public transportation is always dirty and unreliable",
        "Public transportation quality varies by location, funding, and management practices"
    ]
    
    print("Testing single predictions:")
    print("=" * 50)
    
    for text in test_cases:
        data = {"text": text}
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"Text: {result['text'][:60]}...")
                print(f"Biased: {result['is_biased']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Bias Probability: {result['bias_probability']:.3f}")
                print("-" * 50)
            else:
                print(f"Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the API. Make sure the Flask server is running.")
            return

def test_batch_prediction():
    """Test batch prediction"""
    url = f"{BASE_URL}/predict_batch"
    
    texts = [
        "All streaming services are overpriced and offer the same content",
        "Streaming service value varies based on content libraries, pricing models, and user preferences",
        "Windows operating system is always slower than Mac OS",
        "Operating system performance depends on hardware specifications, configuration, and specific use cases"
    ]
    
    data = {"texts": texts}
    
    print("\nTesting batch predictions:")
    print("=" * 50)
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            results = response.json()
            for i, result in enumerate(results['predictions']):
                print(f"Text {i+1}: {result['text'][:60]}...")
                print(f"Biased: {result['is_biased']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print("-" * 30)
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the Flask server is running.")

def test_health_check():
    """Test health check endpoint"""
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print(f"Health Check: {result}")
        else:
            print(f"Health check failed: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API.")

if __name__ == "__main__":
    print("Testing Bias Classification API")
    print("Make sure the Flask server is running (python index.py)")
    print()
    
    # Test health check
    test_health_check()
    
    # Test predictions
    test_single_prediction()
    test_batch_prediction()