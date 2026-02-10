"""
SageMaker Inference Script for Fashion-MNIST CNN
Handles model loading and prediction
"""
import json
import numpy as np
import tensorflow as tf
import os

# Class names for Fashion-MNIST
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def model_fn(model_dir):
    """
    Load the model for inference
    
    Args:
        model_dir: Directory where the model is saved
        
    Returns:
        Loaded TensorFlow model
    """
    print(f'Loading model from {model_dir}')
    model_path = os.path.join(model_dir, '1')
    model = tf.keras.models.load_model(model_path)
    print('Model loaded successfully')
    return model

def input_fn(request_body, content_type='application/json'):
    """
    Parse and preprocess input data
    
    Args:
        request_body: The request payload
        content_type: Content type of the request
        
    Returns:
        Preprocessed numpy array ready for prediction
    """
    print(f'Processing input with content type: {content_type}')
    
    if content_type == 'application/json':
        data = json.loads(request_body)
        
        # Handle different input formats
        if 'instances' in data:
            input_data = np.array(data['instances'], dtype=np.float32)
        elif 'inputs' in data:
            input_data = np.array(data['inputs'], dtype=np.float32)
        else:
            input_data = np.array(data, dtype=np.float32)
        
        # Ensure correct shape (batch, 28, 28, 1)
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=-1)
        elif len(input_data.shape) == 2:
            input_data = input_data.reshape(-1, 28, 28, 1)
        
        print(f'Input shape: {input_data.shape}')
        return input_data
    else:
        raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(input_data, model):
    """
    Make predictions using the loaded model
    
    Args:
        input_data: Preprocessed input data
        model: Loaded model
        
    Returns:
        Model predictions
    """
    print(f'Making predictions for {input_data.shape[0]} samples')
    predictions = model.predict(input_data, verbose=0)
    print('Predictions completed')
    return predictions

def output_fn(predictions, accept='application/json'):
    """
    Format the predictions for response
    
    Args:
        predictions: Model predictions
        accept: Desired response content type
        
    Returns:
        Formatted response (body, content_type)
    """
    print(f'Formatting output with accept type: {accept}')
    
    if accept == 'application/json':
        results = []
        
        for pred in predictions:
            # Get top prediction
            class_idx = int(np.argmax(pred))
            confidence = float(pred[class_idx])
            
            # Get top 3 predictions
            top_3_idx = np.argsort(pred)[-3:][::-1]
            top_3 = [
                {
                    'class': CLASS_NAMES[int(idx)],
                    'class_index': int(idx),
                    'probability': float(pred[idx])
                }
                for idx in top_3_idx
            ]
            
            result = {
                'predicted_class': CLASS_NAMES[class_idx],
                'class_index': class_idx,
                'confidence': confidence,
                'top_3_predictions': top_3,
                'all_probabilities': {
                    CLASS_NAMES[i]: float(pred[i])
                    for i in range(len(CLASS_NAMES))
                }
            }
            results.append(result)
        
        response = {
            'predictions': results,
            'model': 'fashion-mnist-cnn',
            'version': '1.0'
        }
        
        return json.dumps(response), accept
    else:
        raise ValueError(f'Unsupported accept type: {accept}')
