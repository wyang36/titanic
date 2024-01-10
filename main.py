from preprocess import prep_train_val_data, prep_test_data
import torch
from model import Classifier, train
import pandas as pd
import numpy as np


X_train, X_val, y_train, y_val = prep_train_val_data()

X_train = torch.tensor(X_train.astype(np.float32).values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val.astype(np.float32).values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)

# print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


model = Classifier()
val_acc = train(model, X_train, y_train, X_val, y_val)

print(f'best validation acc is {val_acc}')

#inference
X_test = prep_test_data()
X_test_tensor = torch.tensor(X_test.astype(np.float32).values, dtype=torch.float32)
model.eval()
y_pred = model(X_test_tensor).round()

ids = X_test['PassengerId'].copy().reset_index()
# print(y_pred)
ys = pd.DataFrame(y_pred.detach().numpy(), columns=['Survived']).astype('int64')
# print(ids, ys)
prediction_concat = pd.concat([ids, ys], axis=1)[['PassengerId', 'Survived']]
# print(prediction_concat)
prediction_concat.to_csv('data/pred.csv', index=False)