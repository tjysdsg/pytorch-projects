# Projects
## MNIST


## COMPARE2020 Mask

Code is at [mask](mask)

Main training procedure at [here](mask/__main__.py)

NN models are at [here](mask/nn_models)

Run with `python -m mask` at project root dir


Models tried:

- AlexNet: Can't remember, but pretty low
- DenseNet201: 67.5 UAR using lr=0.02, but after about 100 epochs starts to overfit
- GRU with Attention
- LSTM with Attention: stop changing at 63.1 UAR after around 50 epochs
- ResNet152: Can't remember, but around 65?
- VGG19 with batch normalization: Can't remember, but lower than 67

Optimizer:

- SGD
- Adam: low

Learning Rate:

tried the followings with all different models

- 0.01
- 0.02
- 0.1: not working for most of models, and causes UAR fluctuation

Feature:

- MFCC: little worse than LogFBank
- LogFBank: highest
- SSC
- CQT

