# Normalise range 0-1
import pandas as pd
from sklearn import preprocessing


def Normalise(X):
    r = (0, 1)
    a = X.values
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=r)
    a = min_max_scaler.fit_transform(a)
    X = pd.DataFrame(a, columns=X.columns, index=X.index)
    return X

def Norm(X_train, y_train, X_test, y_test):
    range = (0, 1)
    trainX_a = X_train.values
    testX_a = X_test.values
    trainy_a = y_train.values
    testy_a = y_test.values

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=range)
    min_max_scaler.fit(trainX_a)
    min_max_scaler.fit(trainy_a)

    X_train = pd.DataFrame(trainX_a, columns=X_train.columns)
    y_train = pd.DataFrame(trainy_a, columns=y_train.columns)

    X_test = pd.DataFrame(min_max_scaler.transform(testX_a), columns=X_test.columns)
    y_test = pd.DataFrame(min_max_scaler.transform(testy_a), columns=y_test.columns)
    return X_train, y_train, X_test, y_test


def Norm0(X_train, y_train, X_test, y_test):
    range = (0, 1)
    trainX_a = X_train.values
    testX_a = X_test.values
    trainy_a = y_train.values
    testy_a = y_test.values

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=range)
    min_max_scaler.fit_transform(trainX_a)
    X_train = pd.DataFrame(trainX_a, columns=X_train.columns)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=range)
    min_max_scaler.fit_transform(trainy_a)
    y_train = pd.DataFrame(trainy_a, columns=y_train.columns)

    # refitting for now
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=range)
    min_max_scaler.fit(testX_a)
    X_test = pd.DataFrame(min_max_scaler.transform(testX_a), columns=X_test.columns)
    min_max_scaler.fit(testy_a)
    y_test = pd.DataFrame(min_max_scaler.transform(testy_a), columns=y_test.columns)
    return X_train, y_train, X_test, y_test