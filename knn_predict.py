from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib

Local_X = pd.read_csv("X.csv")
Local_y = pd.read_csv("y.csv")['class']


def Euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i])**2
    return np.sqrt(distance)


def Get_Neighbors(train, test_row, num):
    distance = list()  # []
    data = []
    for i in train:
        dist = Euclidean_distance(test_row, i)
        distance.append(dist)
        data.append(i)
    distance = np.array(distance)
    data = np.array(data)
    # Finding the index in ascending order
    index_dist = distance.argsort()
    # Arranging data according to index
    data = data[index_dist]
    # slicing k value from number of data
    neighbors = data[:num]
    return neighbors


def predict_classification(train, test_row, num):
    Neighbors = Get_Neighbors(train, test_row, num)
    Classes = []
    for i in Neighbors:
        Classes.append(i[-1])
    prediction = max(Classes, key=Classes.count)
    return prediction


def accuracy(y_true, y_pred):
    n_correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            n_correct += 1
    acc = n_correct/len(y_true)
    return acc


def vector_predict(image):
    x = np.array(Local_X)
    y = np.array(Local_y)
    si = np.random.permutation(x.shape[0])
    x = x[si]
    y = y[si]
    digit_image = image.reshape(28, 28)
    trainx = x[:2000]
    trainy = y[:2000]
    train = np.insert(trainx, 784, trainy, axis=1)
    prediction = predict_classification(train, train[1244], 3)
    some_digit = train[1244][:-1]
    some_digit_image = some_digit.reshape(28, 28)

    return prediction
