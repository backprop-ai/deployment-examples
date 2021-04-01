import random
import dill

# Our mock model is just a function
def generation_model(input_text, min_length, vocabulary):
    output = [input_text]
    for _ in range(min_length):
        output.append(random.choice(vocabulary))

    return " ".join(output)

# Serialize the model object into a file
with open("mymodel.pickle", "wb") as f:
    dill.dump(generation_model, f, recurse=True)