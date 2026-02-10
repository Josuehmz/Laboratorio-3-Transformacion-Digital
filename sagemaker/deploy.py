"""
AWS SageMaker Deployment Script
Trains and deploys Fashion-MNIST CNN to SageMaker endpoint
"""
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker import get_execution_role
import time
import argparse

# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================
REGION = 'us-east-1'  # Change to your AWS region
BUCKET_NAME = 'sagemaker-fashion-mnist-lab3'  # Change to unique bucket name
PROJECT_NAME = 'fashion-mnist-cnn'

# ============================================================================

def create_s3_bucket(bucket_name, region):
    """Create S3 bucket if it doesn't exist"""
    s3_client = boto3.client('s3', region_name=region)
    
    try:
        print(f'Creating S3 bucket: {bucket_name}...')
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f'✅ Bucket created: {bucket_name}')
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        print(f'✅ Bucket already exists: {bucket_name}')
    except s3_client.exceptions.BucketAlreadyExists:
        print(f'❌ Bucket name already taken globally. Please choose another name.')
        raise
    except Exception as e:
        print(f'❌ Error creating bucket: {e}')
        raise

def get_iam_role():
    """Get IAM role for SageMaker"""
    try:
        # Try to get execution role (works in SageMaker notebooks)
        role = get_execution_role()
        print(f'Using execution role: {role}')
    except:
        # Manual role specification (for local execution)
        sts_client = boto3.client('sts')
        account_id = sts_client.get_caller_identity()['Account']
        role = f'arn:aws:iam::{account_id}:role/SageMakerExecutionRole'
        print(f'Using manual role: {role}')
        print(f'⚠️  Make sure this role exists and has SageMaker permissions')
    
    return role

def train_model(epochs=10, batch_size=128, instance_type='ml.m5.xlarge'):
    """
    Train model in SageMaker
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        instance_type: EC2 instance type for training
    """
    print('\n' + '='*60)
    print('STEP 1: Training Model in SageMaker')
    print('='*60 + '\n')
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Get IAM role
    role = get_iam_role()
    
    # Create S3 bucket
    create_s3_bucket(BUCKET_NAME, REGION)
    
    # Configure TensorFlow estimator
    print(f'\nConfiguring training job...')
    print(f'  Instance type: {instance_type}')
    print(f'  Framework: TensorFlow 2.15')
    print(f'  Epochs: {epochs}')
    print(f'  Batch size: {batch_size}')
    
    estimator = TensorFlow(
        entry_point='train.py',
        source_dir='.',
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version='2.15',
        py_version='py310',
        hyperparameters={
            'epochs': epochs,
            'batch-size': batch_size
        },
        base_job_name=PROJECT_NAME,
        output_path=f's3://{BUCKET_NAME}/models',
        code_location=f's3://{BUCKET_NAME}/code',
        sagemaker_session=sagemaker_session
    )
    
    # Start training
    print(f'\n🚀 Starting training job...')
    print(f'This will take approximately {epochs * 2} minutes...\n')
    
    estimator.fit(wait=True, logs=True)
    
    print(f'\n✅ Training complete!')
    print(f'Model artifacts: {estimator.model_data}')
    
    return estimator

def deploy_endpoint(estimator, instance_type='ml.t2.medium'):
    """
    Deploy model to SageMaker endpoint
    
    Args:
        estimator: Trained SageMaker estimator
        instance_type: EC2 instance type for endpoint
    """
    print('\n' + '='*60)
    print('STEP 2: Deploying Model to Endpoint')
    print('='*60 + '\n')
    
    endpoint_name = f'{PROJECT_NAME}-endpoint-{int(time.time())}'
    
    print(f'Creating endpoint: {endpoint_name}')
    print(f'Instance type: {instance_type}')
    print(f'This will take 5-10 minutes...\n')
    
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        serializer=sagemaker.serializers.JSONSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer()
    )
    
    print(f'\n✅ Endpoint deployed successfully!')
    print(f'Endpoint name: {endpoint_name}')
    print(f'Region: {REGION}')
    
    # Save endpoint name to file
    with open('endpoint_name.txt', 'w') as f:
        f.write(endpoint_name)
    print(f'Endpoint name saved to: endpoint_name.txt')
    
    return predictor, endpoint_name

def test_endpoint(predictor):
    """Test the deployed endpoint with sample predictions"""
    print('\n' + '='*60)
    print('STEP 3: Testing Endpoint')
    print('='*60 + '\n')
    
    import numpy as np
    from tensorflow.keras.datasets import fashion_mnist
    
    # Load a real test image
    print('Loading Fashion-MNIST test data...')
    (_, _), (X_test, y_test) = fashion_mnist.load_data()
    
    # Test with 3 random samples
    indices = np.random.choice(len(X_test), 3, replace=False)
    
    for idx in indices:
        test_image = X_test[idx].reshape(1, 28, 28, 1).astype('float32') / 255.0
        true_label = y_test[idx]
        
        print(f'\nTest sample {idx}:')
        print(f'  True label: {true_label}')
        
        # Make prediction
        result = predictor.predict({'instances': test_image.tolist()})
        
        pred = result['predictions'][0]
        print(f'  Predicted: {pred["predicted_class"]} (confidence: {pred["confidence"]:.4f})')
        print(f'  Top 3 predictions:')
        for i, top_pred in enumerate(pred['top_3_predictions'], 1):
            print(f'    {i}. {top_pred["class"]}: {top_pred["probability"]:.4f}')
    
    print('\n✅ Endpoint is working correctly!')

def main():
    parser = argparse.ArgumentParser(description='Deploy Fashion-MNIST CNN to SageMaker')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--train-instance', type=str, default='ml.m5.xlarge', 
                        help='Instance type for training')
    parser.add_argument('--endpoint-instance', type=str, default='ml.t2.medium',
                        help='Instance type for endpoint')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing model')
    
    args = parser.parse_args()
    
    print('='*60)
    print('AWS SageMaker Deployment Pipeline')
    print('Fashion-MNIST CNN Model')
    print('='*60)
    print(f'Region: {REGION}')
    print(f'S3 Bucket: {BUCKET_NAME}')
    print(f'Project: {PROJECT_NAME}')
    print('='*60)
    
    try:
        # Step 1: Train model
        if not args.skip_training:
            estimator = train_model(
                epochs=args.epochs,
                batch_size=args.batch_size,
                instance_type=args.train_instance
            )
        else:
            print('\n⚠️  Skipping training (using existing model)')
            # Note: You need to load the estimator from a previous training job
            raise NotImplementedError('--skip-training requires loading existing model')
        
        # Step 2: Deploy model
        predictor, endpoint_name = deploy_endpoint(
            estimator,
            instance_type=args.endpoint_instance
        )
        
        # Step 3: Test endpoint
        test_endpoint(predictor)
        
        # Success message
        print('\n' + '='*60)
        print('✅ DEPLOYMENT COMPLETE!')
        print('='*60)
        print(f'\nEndpoint details:')
        print(f'  Name: {endpoint_name}')
        print(f'  Region: {REGION}')
        print(f'  Instance: {args.endpoint_instance}')
        print(f'\n📊 View in AWS Console:')
        print(f'  https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/endpoints/{endpoint_name}')
        print(f'\n💰 Cost estimate:')
        print(f'  Endpoint ({args.endpoint_instance}): ~$0.065/hour = ~$47/month')
        print(f'\n⚠️  IMPORTANT: Delete endpoint to avoid charges:')
        print(f'  python cleanup.py')
        print(f'  OR')
        print(f'  aws sagemaker delete-endpoint --endpoint-name {endpoint_name}')
        print('='*60)
        
    except Exception as e:
        print(f'\n❌ Deployment failed: {e}')
        raise

if __name__ == '__main__':
    main()
