import torch
import torch.nn as nn
import random
import dill

class MyModel(nn.Module):
    def __init__(self):
        # Can't use super() calls
        nn.Module.__init__(self)
        self.layer = nn.Linear(10, 2)

    def classify_text(self, text, labels):
        return random.choice(labels)

    def forward(self, x):
        # Do something with input
        output = self.layer(torch.rand(10))
        # Random output for demonstartion
        return output

    def __call__(self, params, task="text-vectorisation"):
        if task == "text-vectorisation":
            # Input according to Backprop's task specification
            text = params.get("text")
            # Output must be json serializable
            return self.forward(text).tolist()
        elif task == "text-classification":
            # Input according to Backprop's task specification
            text = params.get("text")
            labels = params.get("labels")

            return self.classify_text(text, labels)
        elif task == "custom":
            # Custom task supports any parameter
            my_input = params.get("some_parameter")

            return my_input
        else:
            raise ValueError("Unsupported task!")

# Make model object
# Model must be on cpu
model = MyModel().to("cpu")

# Verify it works
print(model({"some_parameter": "Example 123"}, task="custom"))

# Set the below line to save dependencies!
dill.settings["recurse"] = True
# Serialize the model object into a file called model.bin
with open("model.bin", "wb") as f:
    dill.dump(model, f)