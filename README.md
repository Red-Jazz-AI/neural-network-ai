# nerual-network-ai
A simple AI trained to recognize numbers given in a image made using PyTorch
Protoype for future learning on [PyTorch](https://pytorch.org/)

# PyTorch
PyTorch is a fully featured framework for building deep learning models,
which is a type of machine learning that's commonly used in applications like image recognition and language processing.
Written in Python, it's relatively easy for most machine learning developers to learn and use.


# Installation 
* Go to [PyTorch](https://pytorch.org/) and you should see a installation area, simply choose the settings that fit you best, and use the pip command that is given.
* Then clone this repository or download it.

# Usage
First make sure to train the model

```python

if __name__ == "__main__":
    #Predictions()
    Train()
    #* Pick which one you want to do, usually do train first.

```

make sure for your first time that Train() is run.
After training the model, comment out the Train() and un-comment the predictions.
then simply run main.py


# Expected Results
## Train function
Results:
 ```bash
Epoch: 0 loss is 0.020522331818938255
Epoch: 1 loss is 0.007157251238822937
Epoch: 2 loss is 0.002288956893607974
Epoch: 3 loss is 0.00022219565289560705
Epoch: 4 loss is 0.00017017428763210773
Epoch: 5 loss is 1.2876761502411682e-05
Epoch: 6 loss is 2.205330247306847e-06
Epoch: 7 loss is 5.923189974055276e-07
Epoch: 8 loss is 4.0232922060567944e-07
Epoch: 9 loss is 1.362807051918935e-05
```
All numbers and subject to change and will never be the same

## Predictions Function
```bash
tensor(2, device='cuda:0')
tensor(0, device='cuda:0')
tensor(9, device='cuda:0')
```
the first number is the number on the image.
