"""
SageMaker Training Script for Fashion-MNIST CNN
"""
import argparse
import os
import json
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def create_model():
    """Create CNN model architecture"""
    model = Sequential([
        Input(shape=(28, 28, 1)),
        
        # Conv Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        MaxPooling2D((2, 2)),
        
        # Conv Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        MaxPooling2D((2, 2)),
        
        # Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_data():
    """Load and preprocess Fashion-MNIST data"""
    print('Loading Fashion-MNIST dataset...')
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Normalize and reshape
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f'Training samples: {X_train.shape[0]}')
    print(f'Test samples: {X_test.shape[0]}')
    
    return (X_train, y_train), (X_test, y_test)

def train_model(model_dir, hyperparameters):
    """Main training function"""
    # Get hyperparameters
    epochs = int(hyperparameters.get('epochs', 10))
    batch_size = int(hyperparameters.get('batch_size', 128))
    
    print(f'\nHyperparameters:')
    print(f'  Epochs: {epochs}')
    print(f'  Batch size: {batch_size}')
    
    # Load data
    (X_train, y_train), (X_test, y_test) = load_data()
    
    # Create model
    print('\nCreating model...')
    model = create_model()
    model.summary()
    
    # Train model
    print('\nStarting training...')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate
    print('\nEvaluating model...')
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f'\n{"="*50}')
    print(f'Final Results:')
    print(f'  Test Accuracy: {test_acc:.4f}')
    print(f'  Test Loss: {test_loss:.4f}')
    print(f'{"="*50}\n')
    
    # Save model in TensorFlow SavedModel format
    model_path = os.path.join(model_dir, '1')
    print(f'Saving model to {model_path}...')
    tf.saved_model.save(model, model_path)
    
    # Save training metrics
    metrics = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'train_accuracy': float(history.history['accuracy'][-1]),
        'train_loss': float(history.history['loss'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]),
        'val_loss': float(history.history['val_loss'][-1])
    }
    
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'Metrics saved to {metrics_path}')
    print('Training complete!')
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    
    args, _ = parser.parse_known_args()
    
    hyperparameters = {
        'epochs': args.epochs,
        'batch_size': args.batch_size
    }
    
    print('='*60)
    print('SageMaker Training Job - Fashion-MNIST CNN')
    print('='*60)
    
    train_model(args.model_dir, hyperparameters)
