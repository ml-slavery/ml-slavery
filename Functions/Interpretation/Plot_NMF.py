import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from matplotlib import cm
from Functions.Preprocessing.Normalise import Normalise
from sklearn.decomposition import NMF


def Get_H_matrix(X, best_alpha, best_n, best_solver, max_iter, tol, random_state):
    nmf_opt = NMF(init="nndsvd", solver=best_solver, max_iter=max_iter, tol=tol,
                  n_components=best_n, alpha=best_alpha, random_state=random_state)
    nmf_opt.fit_transform(X)
    H = pd.DataFrame(nmf_opt.components_)
    H.columns = list(X.columns.values)
    return H

def Get_W_matrix(X, best_alpha, best_n, best_solver, max_iter, tol, random_state):
    nmf_opt = NMF(init="nndsvd", solver=best_solver, max_iter=max_iter, tol=tol,
                  n_components=best_n, alpha=best_alpha, random_state=random_state)
    W = pd.DataFrame(nmf_opt.fit_transform(X))
    col_names = W.columns
    # col_names = col_names+1
    W.columns = col_names
    return W

def Plot_NMF_all(H,X, save_path, save_name, figsize=(20,20), bottom_adj= 0.3,
                 component_names=['1','2','3','4','5','6']):
    bounds = np.array([0, 0.1, 0.25, 0.5, 0.75, 1])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    ticks = list(range(0, len(H.columns)))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    H.index=component_names
    img = ax.imshow(H, aspect='auto', interpolation='none', norm=norm)
    x_label_list = list(X.columns.values)
    ax.set_xticks(ticks)
    ax.set_xticklabels(x_label_list)

    yticks = list(range(0, len(H.index)))
    ax.set_yticks(yticks)
    ax.set_yticklabels(list(H.index.values))
    fig.colorbar(img)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=bottom_adj, left=0.1)
    plt.tight_layout()
    plt.savefig(str(save_path)+str(save_name), bbox_inches = 'tight', pad_inches = 0.1)
    plt.clf()
    plt.cla()
    plt.close()

def Plot_Basis(H, X, save_path, save_name, bottom_adj):
    color = cm.inferno_r(np.linspace(.2, .8, 20))
    H_trans = pd.DataFrame(Normalise(np.transpose(H)))
    counter = 1

    l = list()
    for i in list(range(0, len(H_trans.columns))):
        l.append('V' + str(i))
    H_trans.columns = l

    d = {}
    colnames = list(X.columns.values)
    plt.figure(figsize=(6, 4))
    for col in range(0, len(H_trans.columns)):
        d[col] = pd.DataFrame(H_trans.iloc[:, col])
        d[col]['Var'] = colnames
        d[col] = d[col].sort_values(d[col].columns.values[0], axis=0, ascending=True)
        d[col] = d[col].iloc[len(H_trans) - 10:len(H_trans), :]

        plt.barh(d[col]['Var'], d[col].iloc[:, 0], color=color)
        plt.xticks(rotation=90)
        # plt.suptitle('Basis ' + str(counter))
        plt.suptitle(' ')
        plt.subplots_adjust(bottom=bottom_adj, left=0.3)
        plt.rcParams["axes.grid"] = False
        plt.tight_layout()
        plt.savefig(str(save_path) + str(save_name) + '_Basis ' + str(counter))
        counter=counter+1
        plt.clf()
        plt.cla()
        plt.close()

