# Backprop Deployment Examples

These are some examples for model deployments on Backprop's platform.

On a high level, all deployment needs is zipping some files and uploading them on our [Dashboard](https://dashboard.backprop.co).

There are three ways of uploading a model:

1. Using `dill` and serializing a valid model.bin file. See [basic-dill](/basic-dill) for a basic example.
2. Writing a valid `inference.py` file. See [basic-file](/basic-file) for a basic example.
3. Using [Backprop's library](https://github.com/backprop-ai/backprop) to finetune and deploy a model.

For framework specific examples, check out:

1. Text classification and other tasks in [PyTorch](/pytorch).
2. Image classification and other tasks in [TensorFlow](/tensorflow).
3. Custom task in [SKLearn](/sklearn).

A valid model zip file can include as many files as you need, but it **needs** 3 required files: `requirements.txt`, `model.bin` or `inference.py`, and `config.json`.

See the examples for valid versions of these files and read our [docs](https://backprop.co/docs/deploying) for more info.
