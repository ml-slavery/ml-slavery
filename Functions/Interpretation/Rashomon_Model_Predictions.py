from Functions.Interpretation.Plot_NMF import Get_W_matrix, Get_H_matrix
from Functions.Preprocessing.Normalise import Normalise
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import NMF
import pandas as pd

def NMF_DT_MF(X, y, max_iter, tol, init, solver, alpha, n_comp, random_state,
              max_features, min_samples_split, max_depth):
    pipe = Pipeline([
        ("nmf", NMF(init=init, solver=solver,
                    max_iter=max_iter, tol=tol,
                    alpha=int(alpha),
                    n_components=int(n_comp),
                    random_state=int(random_state))),
        ("dt", DecisionTreeRegressor(
            random_state=int(random_state),
            min_samples_split=int(min_samples_split[0]),
            max_features=max_features,
            max_depth=int(max_depth)
        ))])
    CV = LeaveOneOut()
    ypred = cross_val_predict(pipe, X, y, cv=CV)
    ypred = pd.DataFrame(ypred)
    ypred.index = y.index
    return ypred


def NMF_DT(X, y, max_iter, tol, init, solver, alpha, n_comp, random_state,
              min_samples_split, max_depth):
    pipe = Pipeline([
        ("nmf", NMF(init=init, solver=solver,
                    max_iter=max_iter, tol=tol,
                    alpha=int(alpha),
                    n_components=int(n_comp),
                    random_state=int(random_state))),
        ("dt", DecisionTreeRegressor(
            random_state=int(random_state),
            min_samples_split=int(min_samples_split[0]),
            max_depth=int(max_depth)
        ))])
    CV = LeaveOneOut()
    ypred = cross_val_predict(pipe, X, y, cv=CV)
    ypred = pd.DataFrame(ypred)
    ypred.index = y.index
    return ypred


def DT_MF(X, y, random_state, min_samples_split, max_depth, max_features):
    pipe = DecisionTreeRegressor(
        random_state=int(random_state),
        min_samples_split=int(min_samples_split[0]),
        max_features=max_features,
        max_depth=int(max_depth))

    CV = LeaveOneOut()
    ypred = cross_val_predict(pipe, X, y, cv=CV)
    ypred = pd.DataFrame(ypred)
    ypred.index = y.index
    return ypred

def DT(X, y, random_state, min_samples_split, max_depth):
    pipe = DecisionTreeRegressor(
        random_state=int(random_state),
        min_samples_split=int(min_samples_split[0]),
        max_depth=int(max_depth))

    CV = LeaveOneOut()
    ypred = cross_val_predict(pipe, X, y, cv=CV)
    ypred = pd.DataFrame(ypred)
    ypred.index = y.index
    return ypred





