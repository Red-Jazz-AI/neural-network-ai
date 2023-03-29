# neural-network-ai
A simple AI trained to recognize numbers given in a image made using PyTorch.
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
    print("What do you want to do with the model?")
    inpt = input("Train || Predictions: ").lower()
    if inpt in ["train", "prediction", "training", "predictions", "1", "2"]:
        if inpt in ["train", "training", "1"]:
            Train()
        if inpt in ["prediction", "predictions", "2"]:
          Predictions()
    
    #* Pick which one you want to do, usually do train first.


```

make sure for your first time that Train() is run.
After training it, use the predictions function
then simply run main.py


# Expected Results
## Train function
Results:
 ```bash
Epoch: 0 loss is 0.020522331818938255
....

Epoch: 9 loss is 1.362807051918935e-05
```
All numbers and subject to change and will never be the same

## Predictions Function
Results:
```bash
tensor(2, device='cuda:0')
tensor(0, device='cuda:0')
tensor(9, device='cuda:0')
```
the first number is the number on the image.
