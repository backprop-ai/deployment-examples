from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import dill

X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, n_classes=2, shuffle=False)

model = RandomForestClassifier()
model.fit(X, y)

sample_input = [X[0]]

class MyModel:
    def __init__(self, model):
        self.labels = ["a", "b"]
        self.model = model

    def classify(self, features):
        probs = self.model.predict_proba([features])[0]
        label_probs = zip(self.labels, probs)
        probs = {k: v for k,v in label_probs}
        return probs

    def __call__(self, params, task="custom"):
        if task == "custom":
            features = params.get("features")
            probs = self.classify(features)
            return probs
        else:
            raise ValueError("Unsupported task!")

my_model = MyModel(model)

# How Backprop will call your model on request to /custom endpoint
request_body = {
    "features": [0.5, 0.3, -0.1, 0.99]
}

print(my_model(request_body, task="custom"))
# > {'a': 0.08, 'b': 0.92}

# Pickle with dill
# Set the below line to save dependencies!
dill.settings["recurse"] = True
# Serialize the model object into a file called model.bin
with open("model.bin", "wb") as f:
    dill.dump(my_model, f)