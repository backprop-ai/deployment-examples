# PyTorch deployment

This example builds a simple neural network. This is wrapped in a callable object that implements the `custom`, `text-classification` and `text-vectorisation` tasks.

`make_model.py` uses dill to serialize the object into `model.bin`. The final upload is a zip file containing `model.bin`, `requirements.txt`, and `config.json`. This can be uploaded on Backprop's [Dashboard](https://dashboard.backprop.co).

The uploaded model can be invoked by making POST requests with the appropriate body to:

- `api.backprop.co/custom`
- `api.backprop.co/text-classification`
- `api.backprop.co/text-vectorisation`.
