# TensorFlow deployment

This example builds a simple neural network. It is built and exported in `make_model.py`.

`inference.py` implements the required two functions `load_model` and `call_model` that define how to load a model and how the model should be called. The `call_model` function correctly follows the task schemas.

`test.py` shows how Backprop uses the model in a production environment.

The upload .zip will contain `inference.py`, `model.h5`, `config.json` and `requirements.txt`. This can be uploaded on Backprop's [Dashboard](https://dashboard.backprop.co).

The uploaded model can be invoked by making POST requests with the appropriate body to:

- `api.backprop.co/custom`
- `api.backprop.co/image-classification`
