import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Functions.Interpretation.Plot_NMF import Get_H_matrix, Get_W_matrix, Plot_NMF_all, Plot_Basis
from Functions.Interpretation.Plot_DT import TreeImpFeaturesDf
from Functions.Preprocessing.Normalise import Normalise
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance

print('Running')

data_path = "Data/"
df = pd.read_csv(data_path + "Data.csv", index_col=False)

# Create index's:
Index = df[['Country', 'Data_year']]
Index_list = []
for i in range(0, 70):
    Index_list.append(str(Index.Country[i]) + '_' + str(Index.Data_year[i]))
    Index_list[i] = Index_list[i].replace(" ", "_")

# Create X and y:
X = df.drop(["SLAVERY", "Country", "Data_year", "Region"], axis=1)
X = Normalise(X)
X.index = Index_list

y = df['SLAVERY']
y = pd.DataFrame(y)
y.index = Index_list
# ===================
rash_thresh = 0.234

nmf_rash_data = pd.read_csv("Results/All_features/NMF_DT_MF_Rash_results.csv")
nmf_rash_data.rename(columns = {"Unnamed: 0": "MAE"}, inplace=True)
nmf_rash_data.sort_values(by= "MAE", axis=0, inplace=True, ascending=True)

df = nmf_rash_data[nmf_rash_data["MAE"]<rash_thresh]
df.reset_index(inplace=True)
dict = df.to_dict()

solver = 'cd'
max_iter = 350
tol = 0.005
init="nndsvd"
cv = LeaveOneOut()
count = 0
component_names = ["Democratic Rule","Armed Conflict", "Physical Security of Women",
                   "Social Inequal. and Discrimination", "Access to Resources",
                   "Religious and Pol. Freedoms"]


save_path = "Outputs/Rashomon_Set/NMF_Component_Analysis/"
for i in range(0, df.shape[0]):
    model_num = count + 1

    alpha = dict['nmf__alpha'].get(i)
    n_comp = dict['nmf__n_components'].get(i)
    random_state = dict['dt__random_state'].get(i)
    min_samples_split = dict['dt__min_samples_split'].get(i),
    max_features = dict['dt__max_features'].get(i),
    max_depth = dict['dt__max_depth'].get(i)

    # # check cross_val_score :
    # pipe = Pipeline([
    #     ("nmf", NMF(init=init, solver=solver,
    #                 max_iter=max_iter, tol=tol,
    #                 alpha=alpha,
    #                 n_components=n_comp,
    #                 random_state=random_state)),
    #     ("dt", DecisionTreeRegressor(
    #         random_state=random_state,
    #         min_samples_split=min_samples_split[0],
    #         max_features=max_features[0],
    #         max_depth=max_depth
    #     ))])
    # m1_score = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
    # mae_m1 = abs(m1_score).mean()
    # print("SKLearn MAE score: " + str(mae_m1))

    # are the components stable?
    H = Get_H_matrix(X, best_alpha=alpha,
                     best_n=n_comp,
                     random_state=random_state,
                     best_solver=solver, max_iter=max_iter, tol=tol)
    H.to_csv(save_path+str('H_matrix_'+str(i)+'.csv'))

    W = Get_W_matrix(X, best_alpha=alpha,
                     best_n=n_comp,
                     random_state=random_state,
                     best_solver=solver, max_iter=max_iter, tol=tol)
    W.to_csv(save_path+str('W_matrix_'+str(i)+'.csv'))


    Plot_NMF_all(H, X, save_path=save_path, save_name='All_NMF_Components'+str(i)+'.png',
                 figsize=(20, 6), bottom_adj=0.4, component_names=component_names)

    Plot_Basis(H, X, bottom_adj=0.1, save_path=save_path, save_name='R_'+str(i))

    # are the component importances stable?

    # gini:
    Important_Features = TreeImpFeaturesDf(W, y,
                                           best_max_depth=max_depth,
                                           best_tree_seed=random_state,
                                           best_max_features=max_features[0],
                                           best_min_samples_split=min_samples_split[0],
                                           names=True,
                                           component_names=component_names,
                                           save_path=save_path,
                                           save_name='Component_Importance_R'+str(i)+'.png')
    # permutations:
    tree = DecisionTreeRegressor(
        random_state=random_state,
        min_samples_split=min_samples_split[0],
        max_features=max_features[0],
        max_depth=max_depth)

    tree.fit(W, y)
    Permutation_Importance = permutation_importance(tree, W, y, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
    imp = pd.DataFrame(Permutation_Importance.importances_mean)
    comp_name = pd.DataFrame(component_names)
    Permutation_Importance = pd.concat([comp_name, imp], axis=1, ignore_index=True)
    Permutation_Importance.columns = ['Basis', "Model_" + str(model_num)]


    Important_Features.columns = ["Basis_num", "Model_" + str(model_num), "Basis"]
    Important_Features = Important_Features[["Basis", "Model_" + str(model_num)]]
    if count == 0:
        All_Important_Features = Important_Features
        All_Permutation_Features = Permutation_Importance
    else:
        All_Important_Features = All_Important_Features.merge(Important_Features,
                                                              on='Basis', how='left')
        All_Permutation_Features = All_Permutation_Features.merge(Permutation_Importance,
                                                              on='Basis', how='left')
    count = count+1



# --------------------------------------------------------------------

All_Important_Features.to_csv(save_path+'All_Rashomon_Feature_Importance_Gini.csv')
All_Permutation_Features.to_csv(save_path+'All_Rashomon_Feature_Importance_Permutations.csv')

# plot gini
Melt=All_Important_Features.melt(id_vars=['Basis'],var_name='Model',value_name='Importance')
plt.figure(figsize=(20,15))
sns.set_palette("Set2")

chart = sns.catplot(
    x='Basis',
    y='Importance',
    data=Melt,
    kind='bar',
    hue='Model',
    legend=False)

chart.set(xlabel='Basis', ylabel='Importance')
chart.set_xticklabels(rotation=45, horizontalalignment='right', size=7)
chart.ax.legend(loc=1)
# chart._legend.set_title("Model")
plt.subplots_adjust(bottom=0.35)
plt.tight_layout()
plt.savefig(save_path+'All_Rashomon_Feature_Importance_Gini.png')


# plot permutations
Melt=All_Permutation_Features.melt(id_vars=['Basis'],var_name='Model',value_name='Importance')
plt.figure(figsize=(20,15))
sns.set_palette("Set2")

chart = sns.catplot(
    x='Basis',
    y='Importance',
    data=Melt,
    kind='bar',
    hue='Model',
    legend=False)

chart.set(xlabel='Basis', ylabel='Permutation Importance')
chart.set_xticklabels(rotation=45, horizontalalignment='right', size=7)
chart.ax.legend(loc=1)
# chart._legend.set_title("Model")
plt.subplots_adjust(bottom=0.35)
plt.tight_layout()
plt.savefig(save_path+'All_Rashomon_Feature_Importance_Permutations.png')


# ~plot to match~:
#todo: get plot to match predictons
All_Permutation_Features.columns = ["Basis","Best Model: MAE=0.227","Rashomon 2: MAE=0.228","Rashomon 3: MAE=0.230","Rashomon 4: MAE=0.231"]
Melt2=All_Permutation_Features.melt(id_vars=['Basis'],var_name='Model',value_name='Importance')

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 5})

plt.figure(figsize=(25,15))
sns.set_palette("Set2")

chart = sns.catplot(
    x='Basis',
    y='Importance',
    aspect=1.2,
    dodge=True,
    data=Melt2,
    kind='bar',
    hue='Model',
    legend=False,
    palette={"Best Model: MAE=0.227": "orange",
             "Rashomon 2: MAE=0.228": "mediumturquoise",
             "Rashomon 3: MAE=0.230": "grey",
             "Rashomon 4: MAE=0.231" : "tan"
})

plt.legend(loc='upper right', title="", prop={'size': 10})
chart.set(xlabel='', ylabel='Permutation Importance (MAE)')
chart.set_xticklabels(rotation=45, horizontalalignment='right', size=7)
plt.xticks(fontsize=10)
plt.subplots_adjust(bottom=0.4)
plt.tight_layout()
plt.savefig(save_path+'NEW2_All_Rashomon_Feature_Importance_Permutations.png')
plt.clf()
plt.cla()
plt.close()

print ('done!')