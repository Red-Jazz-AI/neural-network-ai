# Import dependencies
from __cuda__ import _isCuda
import torch
import os
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor



#* Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
#* 1,28,28 - classes 0-9


#! Image Classifier Nerual Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)


        )


    def forward(self, x):
        return self.model(x)
    
#* Instance of nerual network, loss, optimizer
clf = ImageClassifier().to('cuda' if _isCuda else "cpu")
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

#* Training flow
def Predictions():
    with open('model_state.pt', "rb") as f:
        clf.load_state_dict(load(f))


    for file in (os.listdir("images")):

      img = Image.open(f"./images/{file}")
      img_tensor = ToTensor()(img).unsqueeze(0).to('cuda' if _isCuda else "cpu")

      print(torch.argmax(clf(img_tensor)))
      

def Train():
    for epoch in range(10): #** train for 10 epochs
        for batch in dataset:
            X,y = batch
            X, y = X.to('cuda' if _isCuda else "cpu"), y.to('cuda' if _isCuda else "cpu")
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            #* Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch: {epoch} loss is {loss.item()}")
    with open('model_state.pt', "wb") as f:
        save(clf.state_dict(), f)


if __name__ == "__main__":
    print("What do you want to do with the model?")
    inpt = input("Train || Predictions: ").lower()
    if inpt in ["train", "prediction", "training", "predictions", "1", "2"]:
        if inpt in ["train", "training", "1"]:
            Train()
        if inpt in ["prediction", "predictions", "2"]:
          Predictions()
    
    #* Pick which one you want to do, usually do train first..
