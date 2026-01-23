# ==========================================
# Lambda Function 1: serializeImageData
# Purpose: Downloads image from S3 and converts to base64 for processing
# ==========================================
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')
    
    # Read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }
# ==========================================
# Lambda Function 2: classifyImage
# Purpose: Sends image to SageMaker endpoint for classification
# ==========================================
import json
import boto3
import base64

ENDPOINT = "image-classification-2025-08-18-02-42-45-019" 

def lambda_handler(event, context):
    
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime')

    if "body" in event:
        body = event["body"]
        if isinstance(body, str):
            body = json.loads(body)
        image_data = body["image_data"]
    else:
        image_data = event["image_data"]
    
    # Decode the image data
    image = base64.b64decode(image_data)
    
    # Make a prediction using boto3 directly
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='image/png',
        Body=image
    )
    
    # Parse the response
    inferences = response['Body'].read().decode('utf-8')
    
    # Return the data back to the Step Function    
    event["inferences"] = inferences
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

# ==========================================
# Lambda Function 3: filterInferences
# Purpose: Filters predictions based on confidence threshold - fails loudly if below threshold
# ==========================================
import json


THRESHOLD = .93

def lambda_handler(event, context):
    # Grab the inferences from the event
    inferences = json.loads(event["inferences"])
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(x > THRESHOLD for x in inferences)


    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        # Pass the event forward
        return {
            'statusCode': 200,
            'body': json.dumps(event)
        }
    else:
        # Force Step Function to fail loudly
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")