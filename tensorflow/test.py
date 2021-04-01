# This is how Backprop will use the model

from inference import load_model, call_model
import os

# Argument is the directory which holds your unzipped files
model = load_model(os.getcwd())

# How Backprop will call your model on request to /image-classification endpoint
request_body = {
    "image": "mock_base64_encoded_image"
}

# Uses your call function that gets passed the model, request_body and task
print(call_model(model, request_body, task="image-classification"))