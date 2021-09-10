import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from datetime import datetime
from sklearn.model_selection import cross_val_predict



def Treefit_names(X, y, best_max_depth, best_min_samples_split, best_max_features, best_tree_seed,
            save_path, save_name, comp_names):

    tree_opt = DecisionTreeRegressor(random_state=best_tree_seed, max_depth=best_max_depth,
                                     min_samples_split=best_min_samples_split, max_features=best_max_features)
    tree_opt.fit(X, y)

    tree.export_graphviz(tree_opt, out_file=str(save_path)+str(save_name)+".dot",
                                    feature_names=comp_names,
                                    filled=True,
                                    special_characters=True,
                                    precision=1,
                                    impurity=False,
                                    node_ids=True,
                                    rounded=True)

    graph = pydotplus.graph_from_dot_file(str(save_path)+str(save_name)+".dot")
    graph.write_png(str(save_path)+str(save_name)+".png")

    return graph


def TreeImpFeaturesDf(X, y, best_max_depth, best_min_samples_split,
                        best_max_features, best_tree_seed, names, component_names,
                      save_path, save_name):
    tree_opt = DecisionTreeRegressor(random_state=best_tree_seed,
                                     max_depth=best_max_depth,
                                     min_samples_split=best_min_samples_split,
                                     max_features=best_max_features)
    tree_opt.fit(X, y)
    important_features = pd.DataFrame(data=tree_opt.feature_importances_, index=X.columns)
    important_features['Basis'] = important_features.index
    important_features = important_features[['Basis',0]]
    important_features.columns = ["Basis", "Importance"]
    if names == True:
        important_features["Basis_name"] = component_names
    important_features.to_csv(save_path+save_name)
    return important_features


def TreeImpFeaturesPlot(X, fig_save_path, fig_save_name, X_col, y_col):
    # X = X.sort_values(by=X_col)
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid")
    sns.set(font_scale=1)

    chart = sns.barplot(
        data=X,
        x=y_col,
        y=X_col,
        palette='Set2',
        orient='h'
    )
    # chart.set(xlim=(0, 1))
    chart.set(xlabel='Decision tree variable importance (0-1)', ylabel='')
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.tight_layout()
    plt.savefig(str(fig_save_path)+fig_save_name)
    return chart


def TreeCVPredict(X, y, best_max_depth, best_min_samples_split, best_max_features,
                  best_tree_seed
                  ):
    y.columns=['Actual']
    tree = DecisionTreeRegressor(random_state=best_tree_seed,
                                     max_depth=best_max_depth,
                                     min_samples_split=best_min_samples_split,
                                     max_features=best_max_features)

    ypred = cross_val_predict(tree, X, y, cv=3)
    ypred = pd.DataFrame(ypred)
    ypred.index = y.index
    return ypred



