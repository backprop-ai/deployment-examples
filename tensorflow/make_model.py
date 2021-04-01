import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import random
import dill

# Everything must be contained in your object
# No external references other than installed modules
class MyModel(tf.keras.Model):
    def __init__(self):
        tf.keras.Model.__init__(self)
        self.labels = ["a", "b", "c", "d", "e"]
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(1, 28*28)),
            tf.keras.layers.Dense(128, activation="relu"),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(5)
        ])
        self.model.compile()

    def classify_image(self, image):
        # Image just random noise
        image = tf.random.normal(shape=(1, 28*28))
        output = self.model(image).numpy()
        # Output must be json serializable
        output = tf.nn.softmax(output).numpy()[0]
        label_probs = zip(self.labels, output)
        probs = {k: v for k,v in label_probs}
        return probs

    def __call__(self, params, task="image-classification"):
        if task == "image-classification":
            # Input according to Backprop's task specification
            # Image must be base64 encoded, but we'll ignore it
            image = params.get("image")
            return self.classify_image(image)
        elif task == "custom":
            # Custom task supports any parameter
            my_input = params.get("some_parameter")

            return my_input
        else:
            raise ValueError("Unsupported task!")

# Make model object
model = MyModel()

# Example local usage
print(model({"image": "mock_base64_encoded_image"}, task="image-classification"))
# > {'a': 0.84332234, 'b': 0.003989996, 'c': 0.0034425238, 'd': 0.017477026, 'e': 0.13176814}

print(model({"some_parameter": "Example 123"}, task="custom"))
# > "Example 123"

# Set the below line to save dependencies!
dill.settings["recurse"] = True

# Serialize the model object into a file called model.bin
with open("model.bin", "wb") as f:
    dill.dump(model, f)

with open("model.bin", "rb") as f:
    model = dill.load(f)
    print(model({"image": "mock_base64_encoded_image"}, task="image-classification"))