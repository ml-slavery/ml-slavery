from Functions.Interpretation.Plot_NMF import Get_W_matrix, Get_H_matrix
from Functions.Preprocessing.Normalise import Normalise
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def NMF_DT_MF(X, y, row, max_iter, tol, n_repeats):
    d={}
    H = Get_H_matrix(X=X, best_alpha=row['nmf__alpha'],
                     best_n=row['nmf__n_components'],
                     best_solver=row['nmf__solver'],
                     random_state=row['dt__random_state'],
                     max_iter=max_iter, tol=tol)
    W = Get_W_matrix(X=X, best_alpha=row['nmf__alpha'],
                     best_n=row['nmf__n_components'],
                     best_solver=row['nmf__solver'],
                     random_state=row['dt__random_state'],
                     max_iter=max_iter, tol=tol)
    m = DecisionTreeRegressor(
        random_state=row['dt__random_state'],
        min_samples_split=row['dt__min_samples_split'],
        max_features=row['dt__max_features'],
        max_depth=row['dt__max_depth'])

    m.fit(W, y)
    r = permutation_importance(m, W, y, n_repeats=n_repeats, random_state=0,
                               scoring="neg_mean_absolute_error")
    imp = pd.DataFrame(r.importances_mean)
    comp = pd.DataFrame(W.columns.values)
    perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
    perm_imp.columns = ['Basis_num', 'Permutation_Importance']

    # Normalise the loadings to be between 0 and 1:
    H = Normalise(H.transpose())

    for index, row in H.iterrows():
        vimp = row * perm_imp['Permutation_Importance']
        if index in d:
            d[index].append(vimp.sum())
        else:
            d[index] = vimp.sum()
    return d

def NMF_DT(X, y, row, max_iter, tol, n_repeats):
    d={}
    H = Get_H_matrix(X=X, best_alpha=row['nmf__alpha'],
                     best_n=row['nmf__n_components'],
                     best_solver=row['nmf__solver'],
                     random_state=row['dt__random_state'],
                     max_iter=max_iter, tol=tol)
    W = Get_W_matrix(X=X, best_alpha=row['nmf__alpha'],
                     best_n=row['nmf__n_components'],
                     best_solver=row['nmf__solver'],
                     random_state=row['dt__random_state'],
                     max_iter=max_iter, tol=tol)
    m = DecisionTreeRegressor(
        random_state=row['dt__random_state'],
        min_samples_split=row['dt__min_samples_split'],
        max_depth=row['dt__max_depth'])

    m.fit(W, y)
    r = permutation_importance(m, W, y, n_repeats=n_repeats, random_state=0,
                               scoring="neg_mean_absolute_error")
    imp = pd.DataFrame(r.importances_mean)
    comp = pd.DataFrame(W.columns.values)
    perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
    perm_imp.columns = ['Basis_num', 'Permutation_Importance']

    # Normalise the loadings to be between 0 and 1:
    H = Normalise(H.transpose())

    for index, row in H.iterrows():
        vimp = row * perm_imp['Permutation_Importance']
        if index in d:
            d[index].append(vimp.sum())
        else:
            d[index] = vimp.sum()
    return d

def NMF_RF(X, y, row, max_iter, tol, n_repeats):
    d={}
    H = Get_H_matrix(X=X, best_alpha=row['nmf__alpha'],
                     best_n=row['nmf__n_components'],
                     best_solver=row['nmf__solver'],
                     random_state=row['dt__random_state'],
                     max_iter=max_iter, tol=tol)
    W = Get_W_matrix(X=X, best_alpha=row['nmf__alpha'],
                     best_n=row['nmf__n_components'],
                     best_solver=row['nmf__solver'],
                     random_state=row['dt__random_state'],
                     max_iter=max_iter, tol=tol)
    m = RandomForestRegressor(random_state=row['random_state'],
                              min_samples_split=row['min_samples_split'],
                              max_depth=row['ax_depth'],
                              n_estimators=row['n_estimators'])

    m.fit(W, y.values.ravel())
    r = permutation_importance(m, W, y, n_repeats=n_repeats, random_state=0,
                               scoring="neg_mean_absolute_error")
    imp = pd.DataFrame(r.importances_mean)
    comp = pd.DataFrame(W.columns.values)
    perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
    perm_imp.columns = ['Basis_num', 'Permutation_Importance']

    # Normalise the loadings to be between 0 and 1:
    H = Normalise(H.transpose())

    for index, row in H.iterrows():
        vimp = row * perm_imp['Permutation_Importance']
        if index in d:
            d[index].append(vimp.sum())
        else:
            d[index] = vimp.sum()
    return d

def DT_MF(X, y, row, n_repeats):
    d={}
    m = DecisionTreeRegressor(
        random_state=int(row['random_state']),
        min_samples_split=int(row['min_samples_split']),
        max_features=row['max_features'],
        max_depth=int(row['max_depth']))

    m.fit(X, y)
    r = permutation_importance(m, X, y, n_repeats=n_repeats, random_state=0,
                               scoring="neg_mean_absolute_error")
    imp = pd.DataFrame(r.importances_mean)
    comp = pd.DataFrame(X.columns.values)
    perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
    perm_imp.columns = ['Variable', 'Permutation_Importance']

    for index, row in perm_imp.iterrows():
        v = row['Variable']
        if v in d:
            d[v].append(row['Permutation_Importance'])
        else:
            d[v] = row['Permutation_Importance']
    return d

def DT(X, y, row, n_repeats):
    d={}
    m = DecisionTreeRegressor(
        random_state=int(row['random_state']),
        min_samples_split=int(row['min_samples_split']),
        max_depth=int(row['max_depth']))

    m.fit(X, y)
    r = permutation_importance(m, X, y, n_repeats=n_repeats, random_state=0,
                               scoring="neg_mean_absolute_error")
    imp = pd.DataFrame(r.importances_mean)
    comp = pd.DataFrame(X.columns.values)
    perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
    perm_imp.columns = ['Variable', 'Permutation_Importance']

    for index, row in perm_imp.iterrows():
        v = row['Variable']
        if v in d:
            d[v].append(row['Permutation_Importance'])
        else:
            d[v] = row['Permutation_Importance']
    return d

def RF(X, y, row, n_repeats):
    d={}
    m = RandomForestRegressor(random_state=row['random_state'],
                              min_samples_split=row['min_samples_split'],
                              max_depth=row['max_depth'],
                              n_estimators=row['n_estimators'])
    m.fit(X, y.values.ravel())
    r = permutation_importance(m, X, y, n_repeats=n_repeats, random_state=0,
                               scoring="neg_mean_absolute_error")
    imp = pd.DataFrame(r.importances_mean)
    comp = pd.DataFrame(X.columns.values)
    perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
    perm_imp.columns = ['Variable', 'Permutation_Importance']
    for index, row in perm_imp.iterrows():
        v = row['Variable']
        if v in d:
            d[v].append(row['Permutation_Importance'])
        else:
            d[v] = row['Permutation_Importance']
    return d


