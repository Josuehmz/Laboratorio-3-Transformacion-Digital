"""
Test a deployed SageMaker endpoint
"""
import boto3
import json
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import argparse

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def test_endpoint(endpoint_name, region='us-east-1', num_samples=5):
    """
    Test SageMaker endpoint with Fashion-MNIST samples
    
    Args:
        endpoint_name: Name of the SageMaker endpoint
        region: AWS region
        num_samples: Number of samples to test
    """
    print('='*60)
    print('Testing SageMaker Endpoint')
    print('='*60)
    print(f'Endpoint: {endpoint_name}')
    print(f'Region: {region}')
    print('='*60 + '\n')
    
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Load Fashion-MNIST test data
    print('Loading Fashion-MNIST test data...')
    (_, _), (X_test, y_test) = fashion_mnist.load_data()
    
    # Test with random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    correct = 0
    total = 0
    
    for i, idx in enumerate(indices, 1):
        # Prepare image
        test_image = X_test[idx].reshape(1, 28, 28, 1).astype('float32') / 255.0
        true_label_idx = int(y_test[idx])
        true_label = CLASS_NAMES[true_label_idx]
        
        print(f'\n📸 Sample {i}/{num_samples} (Index: {idx})')
        print(f'True label: {true_label} (class {true_label_idx})')
        
        # Prepare request
        payload = json.dumps({'instances': test_image.tolist()})
        
        # Invoke endpoint
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            pred = result['predictions'][0]
            
            predicted_label = pred['predicted_class']
            confidence = pred['confidence']
            
            # Check if correct
            is_correct = (pred['class_index'] == true_label_idx)
            if is_correct:
                correct += 1
                status = '✅ CORRECT'
            else:
                status = '❌ WRONG'
            
            total += 1
            
            print(f'{status}')
            print(f'Predicted: {predicted_label} (confidence: {confidence:.4f})')
            print(f'Top 3 predictions:')
            for j, top_pred in enumerate(pred['top_3_predictions'], 1):
                print(f'  {j}. {top_pred["class"]}: {top_pred["probability"]:.4f}')
            
        except Exception as e:
            print(f'❌ Error invoking endpoint: {e}')
            continue
    
    # Summary
    print('\n' + '='*60)
    print('Test Summary')
    print('='*60)
    print(f'Total samples: {total}')
    print(f'Correct predictions: {correct}')
    print(f'Accuracy: {correct/total*100:.2f}%')
    print('='*60)

def main():
    parser = argparse.ArgumentParser(description='Test SageMaker endpoint')
    parser.add_argument('--endpoint-name', type=str, help='SageMaker endpoint name')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to test')
    
    args = parser.parse_args()
    
    # Try to read endpoint name from file if not provided
    if not args.endpoint_name:
        try:
            with open('endpoint_name.txt', 'r') as f:
                args.endpoint_name = f.read().strip()
            print(f'Using endpoint from file: {args.endpoint_name}\n')
        except FileNotFoundError:
            print('Error: No endpoint name provided and endpoint_name.txt not found')
            print('Usage: python test_endpoint.py --endpoint-name YOUR_ENDPOINT_NAME')
            return
    
    test_endpoint(args.endpoint_name, args.region, args.num_samples)

if __name__ == '__main__':
    main()
