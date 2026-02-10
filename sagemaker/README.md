# AWS SageMaker Deployment

Deploy the Fashion-MNIST CNN model to AWS SageMaker endpoint.

## 📋 Prerequisites

1. **AWS Account** with permissions for SageMaker
2. **AWS CLI** configured:
   ```bash
   aws configure
   ```
3. **Python packages**:
   ```bash
   pip install boto3 sagemaker tensorflow
   ```

## 🚀 Quick Start

### 1. Configure

Edit `deploy.py` and update:
```python
REGION = 'us-east-1'  # Your AWS region
BUCKET_NAME = 'your-unique-bucket-name'  # Must be globally unique
```

### 2. Deploy

```bash
cd sagemaker
python deploy.py
```

This will:
- Create S3 bucket
- Train model in SageMaker (10 epochs, ~20 minutes)
- Deploy to endpoint (~10 minutes)
- Test the endpoint

**Total time: ~30 minutes**

### 3. Test Endpoint

```bash
python test_endpoint.py --endpoint-name YOUR_ENDPOINT_NAME
```

Or it will auto-read from `endpoint_name.txt` created during deployment.

### 4. Cleanup (IMPORTANT!)

```bash
python cleanup.py
```

This deletes the endpoint to avoid ongoing charges (~$47/month).

## 📁 Files

- **`train.py`** - Training script for SageMaker
- **`inference.py`** - Inference handler for endpoint
- **`deploy.py`** - Main deployment script
- **`test_endpoint.py`** - Test deployed endpoint
- **`cleanup.py`** - Delete resources
- **`requirements.txt`** - Dependencies

## 💰 Cost Estimate

- **Training** (ml.m5.xlarge): ~$0.23/hour → ~$0.15 per job (10 epochs)
- **Endpoint** (ml.t2.medium): ~$0.065/hour → ~$47/month
- **S3 Storage**: ~$0.023/GB/month → <$1/month

**Total for one-time deployment + 1 hour testing: ~$0.25**

**⚠️ Always delete endpoints when not in use!**

## 🔧 Advanced Options

### Custom training parameters

```bash
python deploy.py --epochs 20 --batch-size 256
```

### Use GPU for training

```bash
python deploy.py --train-instance ml.p3.2xlarge
```

### List all endpoints

```bash
python cleanup.py --list-only
```

### Delete all endpoints

```bash
python cleanup.py --delete-all
```

## 🧪 Using the Endpoint

### Python

```python
import boto3
import json
import numpy as np

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Prepare image (28x28, normalized to [0,1])
test_image = np.random.rand(1, 28, 28, 1).astype('float32')

response = runtime.invoke_endpoint(
    EndpointName='fashion-mnist-cnn-endpoint-1234567890',
    ContentType='application/json',
    Body=json.dumps({'instances': test_image.tolist()})
)

result = json.loads(response['Body'].read())
print(result['predictions'])
```

### AWS CLI

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name fashion-mnist-cnn-endpoint-1234567890 \
  --content-type application/json \
  --body '{"instances": [[[[0.5]]]]}' \
  output.json

cat output.json
```

## 🔍 Monitoring

View endpoint metrics in AWS Console:
```
https://console.aws.amazon.com/sagemaker/home#/endpoints
```

Metrics include:
- Invocations per minute
- Model latency
- CPU/Memory utilization

## 🐛 Troubleshooting

### Error: "Could not assume role"
- Check IAM role permissions
- Ensure role has `AmazonSageMakerFullAccess` policy

### Error: "Bucket already exists"
- Change `BUCKET_NAME` to a unique name

### Error: "Endpoint not found"
- Check region matches
- Verify endpoint is deployed: `python cleanup.py --list-only`

## 📚 References

- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [SageMaker TensorFlow](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/index.html)
- [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
