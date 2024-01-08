import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prep_train_val_data():
    train_X = pd.read_csv('data/train.csv')
    train_y = train_X['Survived'].copy()

    train_X = train_X[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    train_X_encoded = pd.get_dummies(train_X, columns=['Sex'], prefix='Sex')

    X_train, X_val, y_train, y_val = train_test_split(train_X_encoded, train_y, test_size=0.33)

    return X_train, X_val, y_train, y_val


def prep_test_data():
    test_X = pd.read_csv('data/test.csv')
    test_X = test_X[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    test_X_encoded = pd.get_dummies(test_X, columns=['Sex'], prefix='Sex')

    return test_X_encoded