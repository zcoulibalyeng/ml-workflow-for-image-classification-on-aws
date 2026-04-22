# ML Workflow for Scones Unlimited: Image Classification on AWS

An end-to-end machine learning workflow that classifies delivery vehicle types (bicycles vs. motorcycles) using AWS SageMaker, Lambda, and Step Functions. This project demonstrates building scalable, production-ready ML pipelines with monitoring capabilities.

## Project Overview

Scones Unlimited is a scone delivery logistics company that needs to optimize their delivery operations. By automatically classifying delivery vehicles, the system routes bicyclists to nearby orders and motorcyclists to farther destinations—improving efficiency and delivery times.

This project implements:

- A complete ETL pipeline for image data processing
- An image classification model trained on CIFAR-100 data
- Serverless inference workflow using AWS Lambda and Step Functions
- Real-time model monitoring with SageMaker Model Monitor

## Architecture

<img src="arch.svg: width="900px" />

## Project Structure

```
├── starter.ipynb          # Main Jupyter notebook with complete workflow
├── lambda.py              # Lambda function implementations
├── stepfunction.json      # Step Function state machine definition
├── screenshots/           # Step Function execution screenshots
│   └── stepfunction_working.png
└── README.md
```

## Technical Implementation

### 1. Data Staging (ETL)

- **Extract**: Downloaded CIFAR-100 dataset from University of Toronto
- **Transform**: Filtered for bicycle (label 8) and motorcycle (label 48) classes, converted to PNG format
- **Load**: Uploaded processed images and metadata to Amazon S3

### 2. Model Training

- **Algorithm**: AWS SageMaker built-in Image Classification
- **Instance**: ml.p3.2xlarge (GPU-accelerated)
- **Configuration**:
  - Image shape: 3x32x32 (RGB)
  - Classes: 2 (bicycle, motorcycle)
  - Training samples: 1,000
  - Epochs: 30
- **Results**: Achieved ~88% validation accuracy

### 3. Model Deployment

- **Endpoint**: Deployed on ml.m5.xlarge instance
- **Monitoring**: Configured SageMaker Model Monitor with 100% data capture
- **Inference**: Real-time predictions via SageMaker Predictor API

### 4. Serverless Workflow

Three Lambda functions orchestrated by AWS Step Functions:

| Lambda Function | Purpose |
|----------------|---------|
| serializeImageData | Retrieves image from S3 and encodes to base64 |
| classifyImage | Invokes SageMaker endpoint for inference |
| filterInferences | Filters predictions below confidence threshold (93%) |

### 5. Model Monitoring

- Captured inference data stored in S3 for analysis
- Visualization of confidence scores over time
- Production threshold monitoring (94% confidence)

## Technologies Used

- **AWS SageMaker**: Model training, deployment, and monitoring
- **AWS Lambda**: Serverless compute for workflow steps
- **AWS Step Functions**: Workflow orchestration
- **Amazon S3**: Data storage for images and model artifacts
- **Python**: Primary programming language
- **Libraries**: boto3, pandas, numpy, matplotlib, sagemaker SDK

## Key Features

- Scalable architecture supporting high-volume inference requests
- Automated confidence filtering to ensure prediction quality
- Comprehensive monitoring for model drift detection
- Event-driven workflow easily integrable with production systems
- Cost-effective serverless design

## Results

The deployed model achieves:
- Training accuracy: ~96%
- Validation accuracy: ~88%
- Production confidence threshold: 94%

Sample inference output:
```
[0.9898792505264282, 0.010120709426701069]
# Interpretation: 99% bicycle, 1% motorcycle
```

## Monitoring Visualization

The project includes visualization for tracking model performance:
- Confidence distribution across predictions
- Production readiness assessment
- Threshold-based filtering analysis

## Setup and Prerequisites

1. AWS Account with SageMaker access
2. IAM role with permissions for:
   - SageMaker (full access)
   - S3 (read/write)
   - Lambda (execution)
   - Step Functions (execution)
3. SageMaker Studio or Notebook instance (ml.t3.medium recommended)
4. Python 3.8+ environment

## Usage

1. Clone this repository
2. Open `notebook.ipynb` in SageMaker Studio
3. Execute cells sequentially to:
   - Process and upload training data
   - Train the image classification model
   - Deploy the model endpoint
   - Create and test Lambda functions
   - Configure Step Functions workflow
   - Run inference tests and visualize results


