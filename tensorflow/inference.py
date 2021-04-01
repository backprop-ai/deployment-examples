import os
import tensorflow as tf

labels = ["a", "b", "c", "d", "e"]

def load_model(dir_path):
    h5_path = os.path.join(dir_path, "model.h5")
    return tf.keras.models.load_model(h5_path)

def classify_image(model, image):
    global labels
    # Image just random noise
    image = tf.random.normal(shape=(1, 28*28))
    output = model(image).numpy()
    # Output must be json serializable
    output = tf.nn.softmax(output).numpy().tolist()[0]
    label_probs = zip(labels, output)
    probs = {k: v for k,v in label_probs}
    return probs

def call_model(model, params, task="image-classification"):
    if task == "image-classification":
        # Input according to Backprop's task specification
        # Image must be base64 encoded, but we'll ignore it
        image = params.get("image")
        return classify_image(model, image)
    elif task == "custom":
        # Custom task supports any parameter
        my_input = params.get("some_parameter")

        return my_input
    else:
        raise ValueError("Unsupported task!")