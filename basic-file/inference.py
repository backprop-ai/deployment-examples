import os
import dill

vocabulary = ["where", "what", "monkey", "backprop", "power", "is"]

def load_model(dir_path):
    model_path = os.path.join(dir_path, "mymodel.pickle")
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model


def call_model(model, params, task="image-classification"):
    if task == "text-generation":
        global vocabulary
        # Input according to Backprop's task specification
        text = params.get("text")
        min_length = params.get("min_length", 5)
        # Output must be json serializable
        return model(text, min_length, vocabulary)
    else:
        raise ValueError("Unsupported task!")