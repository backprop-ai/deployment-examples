# SKLearn deployment

This example builds a random forest classifier. It is wrapped in a callable object that implements the custom task.

`make_model.py` uses dill to serialize the object into `model.bin`. The final upload is a zip file containing `model.bin`, `requirements.txt`, and `config.json`. This can be uploaded on Backprop's [Dashboard](https://dashboard.backprop.co).

The uploaded model can be invoked by making POST requests with the appropriate body to `api.backprop.co/custom`.
