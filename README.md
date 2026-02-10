# Lab 3: Exploring Convolutional Layers Through Data and Experiments
## By: Josué Hernandez
##  Description

This lab explores **convolutional layers** as fundamental architectural components in neural networks, analyzing how design decisions affect performance, scalability, and interpretability.

**Dataset:** Fashion-MNIST (70,000 clothing images 28×28 in grayscale)

##  Learning Objectives

By completing this lab, you will be able to:

-  Understand the role and mathematical intuition behind convolutional layers
-  Analyze how architectural decisions (kernel size, depth, stride, padding) affect learning
-  Compare convolutional layers with fully connected layers for image-like data
-  Perform a meaningful exploratory data analysis (EDA) for NN tasks
-  Communicate architectural and experimental decisions clearly

##  Lab Structure

The notebook `01_eda_dataset.ipynb` is organized into **5 main tasks** + **bonus**:

### **1. Dataset Exploration (EDA)**
- Dataset size and class distribution
- Image dimensions and channels
- Sample examples per class
- Preprocessing (normalization, reshaping)

### **2. Baseline Model (Non-Convolutional)**
- MLP implementation (Flatten + Dense layers)
- Architecture: 784 → 128 → 64 → 10
- Number of parameters: ~115K
- Performance: ~88-89% test accuracy
- **Observed limitations**: Loss of spatial structure, too many parameters, no translation invariance

### **3. Convolutional Architecture Design**
- CNN designed from scratch (not copied from tutorials)
- **Justified architecture**:
  - 2 convolutional layers (32 and 64 filters)
  - 3×3 kernels with padding='same'
  - MaxPooling 2×2
  - ReLU activation
  - Dropout 0.3
- Number of parameters: ~150K
- Performance: ~91-92% test accuracy

### **4. Controlled Experiments**
- **Experiment**: Kernel size comparison (3×3 vs 5×5)
- **Controlled variables**: Number of layers, filters, pooling, learning rate
- **Quantitative results**:
  - 3×3: ~91% accuracy, ~150K parameters
  - 5×5: ~91% accuracy, ~255K parameters (+70% parameters, marginal improvement)
- **Trade-offs**: Computational efficiency vs receptive field

### **5. Interpretation and Architectural Reasoning** ⭐
**Most important section (heavily graded)**

#### 5.1 Why do convolutions outperform the baseline?
- Preservation of spatial structure
- Parameter sharing
- Local connectivity
- Feature hierarchy
- Better generalization with fewer parameters

#### 5.2 What inductive bias does convolution introduce?
- **Spatial locality**: Nearby pixels are correlated
- **Translation invariance**: A pattern useful in one position is useful anywhere
- **Compositional hierarchy**: Complex features are built from simple ones

#### 5.3 When NOT to use convolutions?
- Tabular data without spatial structure
- Sequences with long-range dependencies
- Graphs with irregular topology
- Signals with irregular periodicity
- Problems requiring global reasoning

### **BONUS: Visualization**
- Learned filters (first convolutional layer)
- Feature maps (intermediate activations)
- Interpretation of what each filter detects

##  Requirements

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- TensorFlow/Keras 2.x or 3.x
- NumPy
- Matplotlib
- Seaborn
- Pandas

##  Execution

1. Clone the repository or download the files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:
   ```bash
   jupyter notebook 01_eda_dataset.ipynb
   ```
4. Run cells sequentially (Runtime → Run All)

**Estimated execution time:** 10-15 minutes (with GPU) / 30-40 minutes (with CPU)

##  Key Results

| Model | Parameters | Test Accuracy | Test Loss |
|-------|-----------|---------------|-----------|
| Baseline MLP | ~115K | ~88-89% | ~0.31 |
| CNN (3×3) | ~150K | ~91-92% | ~0.25 |
| CNN (5×5) | ~255K | ~91% | ~0.26 |

**Conclusion:** CNNs achieve better accuracy with similar complexity to the baseline MLP, demonstrating the effectiveness of convolutional inductive bias for visual data.

## ☁️ Cloud Deployment Attempt

An attempt was made to deploy the trained model to **AWS SageMaker** for production inference. The deployment pipeline includes:

- **Training Script** (`sagemaker/train.py`): SageMaker-compatible training job
- **Inference Handler** (`sagemaker/inference.py`): Real-time prediction endpoint
- **Deployment Pipeline** (`sagemaker/deploy.py`): Automated deployment to SageMaker endpoint
- **Testing Suite** (`sagemaker/test_endpoint.py`): Endpoint validation
- **Cleanup Tools** (`sagemaker/cleanup.py`): Resource management

**Issue:** IAM role permission error prevented successful deployment.

