import random
import dill

class MyModel:
    def __init__(self):
        self.vocabulary = ["where", "what", "monkey", "backprop", "power", "is"]

    def generate_text(self, input_text, min_length):
        output = [input_text]
        for _ in range(min_length):
            output.append(random.choice(self.vocabulary))

        return " ".join(output)

    def __call__(self, params, task="text-generation"):
        if task == "text-generation":
            # Input according to Backprop's task specification
            text = params.get("text")
            min_length = params.get("min_length", 5)
            # Output must be json serializable
            return self.generate_text(text, min_length)
        else:
            raise ValueError("Unsupported task!")

# Make model object
model = MyModel()

# How Backprop will call your model on request to /text-generation endpoint
# Backprop passes the text-generation API parameters to params and sets task as "text-generation"
request_body = {
    "text": "Test",
    "min_length": 3
}

print(model(request_body, task="text-generation"))

# Set the below line to save dependencies!
dill.settings["recurse"] = True
# Serialize the model object into a file called model.bin
with open("model.bin", "wb") as f:
    dill.dump(model, f)