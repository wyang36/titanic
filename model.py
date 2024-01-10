import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import copy

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 60)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        # x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
    

def train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
 
    n_epochs = 2000   # number of epochs to run
    batch_size = 10  # size of each batch
    # batch_start = torch.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
 
    for epoch in range(n_epochs):
        model.train()
       
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        # print progress
        acc_train = (y_pred.round() == y_train).float().mean()
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc_val = (y_pred.round() == y_val).float().mean()
        acc_val = float(acc_val)
        print(f'epoch {epoch}, training accuracy {acc_train}, validation accuracy: {acc_val}')
        if acc_val > best_acc:
            best_acc = acc_val
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc