import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from matplotlib import colors
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import NMF
from Functions.Interpretation.Plot_NMF import Get_H_matrix, Get_W_matrix, \
    Plot_NMF_all, Plot_Basis
from Functions.Interpretation.Plot_DT import Treefit_names, TreeImpFeaturesDf, TreeImpFeaturesPlot
from Functions.Preprocessing.Normalise import Normalise
from sklearn.inspection import plot_partial_dependence, partial_dependence
from mpl_toolkits.mplot3d import Axes3D
from sklearn.inspection import permutation_importance
from pycebox.ice import ice, ice_plot

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
# --------------------------------------------------------------------------

param_dict = {"MAE":
                  {'best_n': 6,
                  'best_alpha': 2,
                  'best_max_depth': 6,
                  'best_min_samples': 3,
                  'best_tree_seed': 45,
                  'best_max_features': 0.3}}

# build best model:

pipe = Pipeline([
        ("nmf", NMF(init="nndsvd", solver="cd", max_iter=350, tol=0.005,
                    alpha=param_dict["MAE"].get('best_alpha'),
                    n_components=param_dict["MAE"].get('best_n'),
                    random_state=param_dict["MAE"].get('best_tree_seed'))),
        ("dt", DecisionTreeRegressor(
            random_state=param_dict["MAE"].get('best_tree_seed'),
            min_samples_split=param_dict["MAE"].get('best_min_samples'),
            max_features=param_dict["MAE"].get('best_max_features'),
            max_depth=param_dict["MAE"].get('best_max_depth')
        ))])

# ---------------------------------------- check cross_val_score :
cv = LeaveOneOut()
m1_score = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
mae_m1 = abs(m1_score).mean()
print("SKLearn MAE score: "+str(mae_m1))

# ---------------------------------------- check baselines :

dummy_regr_mean = DummyRegressor(strategy="mean")
dummy_regr_med = DummyRegressor(strategy="median")

# compare LOO anal performance
score_mean = cross_val_score(dummy_regr_mean, X, y, cv=cv, scoring="neg_mean_absolute_error")
score_med = cross_val_score(dummy_regr_med, X, y, cv=cv, scoring="neg_mean_absolute_error")

mae_meanreg = abs(score_mean).mean()
mae_medreg = abs(score_med).mean()

print("LOO mean baseline: "+str(mae_meanreg))
print("LOO median baseline: "+str(mae_medreg))

# -------------------------------------------- t-tests compared to Baseline:

k2, p = stats.normaltest(m1_score)
alpha = 0.05
print("p = {:f}".format(p))

if p < alpha:
    print("The null hypothesis (that X comes from a normal distribution) can be rejected = not normal distribution")
else:
    print("The null hypothesis (that X is normal) cannot be rejected = is normal")

# non-parametric (not normal) paired (same data) t-test = Wilcoxon Signed-Rank test:

stat, p = stats.wilcoxon(x=m1_score, y=score_mean, zero_method='wilcox', alternative='two-sided')
print("m1 and mean baseline, p = {:f}".format(p))

stat, p = stats.wilcoxon(x=m1_score, y=score_med, zero_method='wilcox', alternative='two-sided')
print("m1 and med baseline, p = {:f}".format(p))


# ===================Interpret best model==================================

save_path = "Outputs/Best_NMF_Model/"

component_names = ["Democratic Rule","Armed Conflict", "Physical Security of Women",
                   "Social Inequal. and Discrimination", "Access to Resources", "Religious and Pol. Freedoms"]


key = "MAE"
# Get components and loadings
W = Get_W_matrix(X, tol=0.005, max_iter=350, best_solver='cd',
                 best_alpha=param_dict[key].get('best_alpha'),
                 best_n=param_dict[key].get('best_n'),
                 random_state=param_dict[key].get('best_tree_seed'))
W.to_csv(save_path+str('H_matrix.csv'))

H = Get_H_matrix(X, tol=0.005, max_iter=350, best_solver='cd',
                 best_alpha=param_dict[key].get('best_alpha'),
                 best_n=param_dict[key].get('best_n'),
                 random_state=param_dict[key].get('best_tree_seed'))
H.to_csv(save_path+str('H_matrix.csv'))

# Plot and interpret Components

Plot_NMF_all(H=H, X=X, save_path=save_path, save_name='NMF_plot_all.png',
             figsize=(25,10), bottom_adj= 0.3)
plt.clf()
Plot_Basis(H=H, X=X, save_path=save_path, save_name='NMF_Basis_', bottom_adj=0.2)
plt.clf()

# Importance of Components

important_features = TreeImpFeaturesDf(W, y,
                                       best_max_depth=param_dict[key].get('best_max_depth'),
                                       best_tree_seed=param_dict[key].get('best_tree_seed'),
                                       best_max_features=param_dict[key].get('best_max_features'),
                                       best_min_samples_split=param_dict[key].get('best_min_samples'),
                                       names=True,
                                       component_names=component_names,
                                       save_path = save_path,
                                       save_name='Gini_Importance_Plot.csv'
                                       )

TreeImpFeaturesPlot(X=important_features, fig_save_path=save_path,
                    fig_save_name='Gini_Importance_Plot.png',
                    y_col='Importance',
                    X_col='Basis_name')

# Random Permuation Importance ----------------------------------------------

tree = DecisionTreeRegressor(
            random_state=param_dict["MAE"].get('best_tree_seed'),
            min_samples_split=param_dict["MAE"].get('best_min_samples'),
            max_features=param_dict["MAE"].get('best_max_features'),
            max_depth=param_dict["MAE"].get('best_max_depth'))

tree.fit(W, y)
r = permutation_importance(tree, W, y, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")

imp = pd.DataFrame(r.importances_mean)
sd_imp = pd.DataFrame(r.importances_std)
comp = pd.DataFrame(W.columns.values)

comp_name = pd.DataFrame(component_names)
perm_imp = pd.concat([comp, comp_name, imp, sd_imp], axis=1, ignore_index=True)
perm_imp.columns = ['Basis_num', 'Basis','Permutation_Importance','SD']

plt.figure(figsize=(8,2))
sns.set(style="whitegrid")
sns.set(font_scale=1)

chart = sns.barplot(
    data=perm_imp,
    x='Permutation_Importance',
    y='Basis',
    color='orange',
    orient='h'
)

chart.set(xlabel='Permutation Importance (MAE)', ylabel='')
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.tight_layout()
plt.savefig(str(save_path) + 'NEW_Permutation_Importances.png')

# Plot tree ------------------------------------------------------------------------

# Standardise W to be between 0 and 1:
Wn = Normalise(W)

Treefit_names(Wn, y,
        best_max_depth=param_dict[key].get('best_max_depth'),
        best_tree_seed=param_dict[key].get('best_tree_seed'),
        best_max_features=param_dict[key].get('best_max_features'),
        best_min_samples_split=param_dict[key].get('best_min_samples'),
        save_path=save_path,
        save_name='DT_Plot',
        comp_names=component_names)

# check and plot correlations between basis -----------------------------------

f = plt.figure(figsize=(12, 10))
cor=round(Wn.corr(),2)
cor.index = component_names
cor.columns = component_names
cor.to_csv(save_path+"Basis_Correlations.csv")

bounds = [-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0]
norm = colors.BoundaryNorm(bounds, ncolors=12)
map = cm.get_cmap("coolwarm")
plt.matshow(cor, fignum=f.number, cmap=map)
plt.xticks(range(Wn.shape[1]), Wn.columns, fontsize=14, rotation=45)
plt.yticks(range(Wn.shape[1]), Wn.columns, fontsize=14)

cb = plt.colorbar(boundaries=bounds, norm=norm)
cb.ax.tick_params(labelsize=14)
plt.subplots_adjust(bottom=0.6, left=0.6)
plt.grid(b=None)
plt.savefig(str(save_path) + 'Basis_Correlation_Matrix.png', bbox_inches='tight')
plt.clf()

# seaborn plot
plt.figure(figsize=(15, 15))
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 5})
g = sns.heatmap(cor,
            xticklabels=cor.columns.values,
            yticklabels=cor.columns.values,
            cbar=False, annot = True)
plt.subplots_adjust(bottom=0.3, left=0.3)
# g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.savefig(str(save_path) + 'NEW Basis_Correlation_Matrix.png', bbox_inches='tight')


# 2x2 PD plot --------------------------------------------------------------
sns.set_style("whitegrid")
new_path = save_path + 'PDP_2way_Interactions/'
fig = plt.figure()

Wn.columns=component_names

f1 = ["Physical Security of Women", "Access to Resources"]
f2 = ["Social Inequality","Armed Conflict"]
f3 = ["Social Inequality","Political Instability"]
f4 = ["Social Inequality","Basic Needs and Human Rights"]

f=f1

pdp, axes = partial_dependence(tree, Wn, features=f, grid_resolution=20)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(f[0])
ax.set_ylabel(f[1])
ax.set_zlabel('Partial dependence')

#  pretty init view
ax.view_init(elev=22, azim=122)
# plt.colorbar(surf)
plt.suptitle('Partial Dependence of Slavery Prevalence on Median\n'
                  '{} and {}'.format(str(f[0]),str(f[1])))
plt.subplots_adjust(bottom=0.3)
plt.savefig(new_path+"manual_3D_{}_and_{}".format(str(f[0]),str(f[1]))+".png")

# get all 2-way interactions UNIQUE pairs on 3D graph ---------------------------

# new_path = path + 'PDP_2way_interactions/'
#
# n = 5
# unique_pairs = (n*(n-1))/2
# print(unique_pairs)
#
# for num, fs in enumerate(combinations(Wn.columns.values, 2)):
#     num = num+1
#     fs = list(fs)
#     print('Plotting interaction: {} '.format(num) + str(fs[0]) + ' * ' + str(fs[1]))
#
#     pdp, axes = partial_dependence(tree, Wn, features=fs, grid_resolution=20)
#     XX, YY = np.meshgrid(axes[0], axes[1])
#     Z = pdp[0].T
#     ax = Axes3D(fig)
#     surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
#                            cmap=plt.cm.BuPu, edgecolor='k')
#     ax.set_xlabel(str(fs[0]))
#     ax.set_ylabel(str(fs[1]))
#     ax.set_zlabel('Partial Dependence')
#
#     #  pretty init view
#     ax.view_init(elev=22, azim=122)
#
#     # plt.colorbar(surf)
#     plt.suptitle('Partial Dependence of Slavery Prevalence on Median\n'
#                  '{} and {}'.format(str(fs[0]), str(fs[1])))
#     plt.subplots_adjust(bottom=0.3)
#     plt.savefig(new_path+'3D_pd_inter_'+str(fs[0])+'_'+str(fs[1])+".png")
#

# 2d heatmap version -------------------------------------:


W = Normalise(W)
W.columns = component_names
tree = tree.fit(W, y)

def Jpred_fun(W):
    pred = tree.predict(W)
    return pred

# sns.set_style("whitegrid")
fig = plt.figure()

f = ["Physical Security of Women", "Access to Resources"]

pdp, axes = partial_dependence(tree, W, features=f, grid_resolution=10)

# ax1 = np.round(np.linspace(0.1,1,49),2)
ax2 = np.round(np.linspace(0.1,1,10),2)
print(ax2)

pdp_df = pd.DataFrame(pdp[0], columns=ax2, index=ax2)

plt.rcParams.update({'font.size': 11})
chart = sns.heatmap(pdp_df, vmin=0, vmax=1.25, linewidths=0, cmap="YlGnBu",
                    # cbar_kws = dict(use_gridspec=False,location="top"))
                    cbar_kws={'label': '\n Partial Dependence'})
chart.set_yticklabels(labels=ax2, va='bottom')
chart.set_xticklabels(labels=ax2, ha='left')
chart.set(xlabel="Physical Security of Women", ylabel='Access to Resources')
chart.set(xlim=(0, 11), ylim=(0,11))
chart.tick_params(bottom=False,left=False)
plt.tight_layout()
plt.savefig(new_path+"manual_2D_{}_and_{}_gridres10".format(str(f[0]),str(f[1]))+".pdf")

# ----------------  JITTER ICE plots the first 3 features:

font = {'size' : 18}
c_names = ["\n Democratic \n Rule","\n Armed \n Conflict", "\n Physical \n Security \n of Women",
                   "\n Social Inequal. \n and \n Discrimination", "\n Access \n to \n Resources", "\n Religious \n and Pol. \n Freedoms"]

plt.rc('font', **font)
fig, axes = plt.subplots(1, 6, figsize=(15, 10))
for i, col in enumerate(W.columns):
    # generate ICE data points
    ice_df = ice(W, column=col, predict=Jpred_fun)

    # for each row in ice_df add jitter:
    ice_dfN = ice_df.apply(lambda x: x + 0.2 * np.random.rand() -0.05, axis=0)

    # plot
    plt.subplot(axes[i])
    ice_plot(ice_dfN, linewidth=1, plot_pdp=True,
                    pdp_kwargs={'c': 'k', 'linewidth': 3}, ax=axes[i])
    axes[i].set_xlabel(c_names[i])
    axes[i].set_ylim(0,2.5,0.5)
    axes[i].set_yticklabels([])
    axes[i].set_xlim(0, 1.1, 0.5)
    axes[i].set_xticklabels(['0', '1'])
    axes[i].tick_params(axis='x', labelsize=15)
    plt.grid(b=False, which='major', axis='both')

    # axes[i].set_xticklabels([])

axes[0].set_ylabel('Predicted Prevalence')
axes[0].set_yticklabels(['0','0.5','1','1.5','2','2.5'])


plt.suptitle(' ', fontsize=18)
plt.tight_layout()
plt.savefig(save_path+'ICE_jitter.pdf')

print('done')

