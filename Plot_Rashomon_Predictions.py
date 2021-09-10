import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.lines import Line2D
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
from Functions.Interpretation.Rashomon_Model_Predictions import NMF_DT_MF, NMF_DT, DT, DT_MF

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
X.rename(index= {'Democratic_Republic_of_the_Congo_2018': 'Dem_Rep_Congo_2018'}, inplace=True)

y = df['SLAVERY']
y = pd.DataFrame(y)
y.index = Index_list
y.rename(index= {'Democratic_Republic_of_the_Congo_2018': 'Dem_Rep_Congo_2018'}, inplace=True)

keep_vars = ['KOF_Globalis','Work_rightCIRI','Trade_open','GDPpc',
 'Armedcon','Poverty', 'Stunting_u5s','Undernourish','Wasting_u5s','Maternal_mort',
 'Neonatal_mort','Literacy_15_24yrs','F_school','ATMs','Child_lab','Unemploy','Infrastruct',
 'Internet_use','Broadband','Climate_chg_vuln','CPI','Minority_rule','Freemv_M',
 'Freemv_F','Free_discuss','Soc_powerdist','Democ',
 'Sexwrk_condom','Sexwrk_Syphilis','AIDS_Orph','Rape_report',
 'Rape_enclave','Phys_secF','Gender_equal']
# ===================
rashomon_threshold = 0.2340852835055713

df = pd.read_csv("Outputs/Rashomon_Set/Rash_Results_all_parms_{}.csv".format(rashomon_threshold), index_col=False)

# Get predictions for all models in rashomon set: ----------------------------------------
solver = 'cd'
max_iter = 350
tol = 0.005
init="nndsvd"
cv = LeaveOneOut()
model_num = 0
count=0
component_names = ["Democratic Rule","Armed Conflict", "Physical Security of Women",
                   "Social Inequal. and Discrimination", "Access to Resources",
                   "Religious and Pol. Freedoms"]

save_path = "Outputs/Rashomon_Set/Plot_Rashomon_Predictions/"

dict = df.to_dict()
for i in range(0, df.shape[0]):
    model_name = dict['Model'].get(i)
    features = dict['Features'].get(i)
    if 'NMF->' in model_name:
        n_comp = dict['nmf__n_components'].get(i)
        random_state = dict['dt__random_state'].get(i)
        alpha = dict['nmf__alpha'].get(i)
        if 'DT' in model_name:
            min_samples_split = dict['dt__min_samples_split'].get(i),
            max_depth = dict['dt__max_depth'].get(i)
            if '(MF)' in model_name:
                max_features = dict['dt__max_features'].get(i)
                if features == "All_features":
                    ypred = NMF_DT_MF(X=X, y=y, max_iter=max_iter, tol=tol,
                                      init=init, solver=solver, alpha=alpha,
                                      n_comp=n_comp, random_state=random_state,
                                      max_features=max_features,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth)
                if features == "Theory_features":
                    X = X[keep_vars]
                    ypred = NMF_DT_MF(X=X, y=y, max_iter=max_iter, tol=tol,
                                      init=init, solver=solver, alpha=alpha,
                                      n_comp=n_comp, random_state=random_state,
                                      max_features=max_features,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth)
            else:
                if features == "All_features":
                    ypred = NMF_DT(X=X, y=y, max_iter=max_iter, tol=tol,
                                      init=init, solver=solver, alpha=alpha,
                                      n_comp=n_comp, random_state=random_state,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth)
                if features == "Theory_features":
                    X = X[keep_vars]
                    ypred = NMF_DT(X=X, y=y, max_iter=max_iter, tol=tol,
                                      init=init, solver=solver, alpha=alpha,
                                      n_comp=n_comp, random_state=random_state,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth)
    else:
        if 'DT' in model_name:
            min_samples_split = dict['min_samples_split'].get(i),
            max_depth = dict['max_depth'].get(i)
            random_state = dict['random_state'].get(i)
            if '(MF)' in model_name:
                max_features = dict['max_features'].get(i)
                if features == "All_features":
                    ypred = DT_MF(X=X, y=y, random_state=random_state,
                                      max_features=max_features,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth)
                if features == "Theory_features":
                    X = X[keep_vars]
                    ypred = DT_MF(X=X, y=y, random_state=random_state,
                                      max_features=max_features,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth)
            else:
                if features == "All_features":
                    ypred = DT(X=X, y=y, random_state=random_state,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth)
                if features == "Theory_features":
                    X = X[keep_vars]
                    ypred = DT(X=X, y=y, random_state=random_state,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth)
    model_num = model_num + 1
    ypred.columns = ['Model_' + str(model_num)]
    y.columns = ['Actual']

    if count == 0:
        all_results=y.join(ypred)
    else:
        all_results=all_results.join(ypred)
    count = count+1

# plot all:
all_results['Country_Year'] = y.index
all_results.to_csv(save_path+'Rashomon_prediction_results.csv', index=False)
all_results.columns = ["Actual","Best Model","Rashomon 1","Rashomon 2","Rashomon 3", "Rashomon 4", "Rashomon 5", "Country_Year"]
all_resultsm=all_results.melt(id_vars=['Country_Year'],var_name='Model',value_name='Prediction')

plt.figure(figsize=(20,17))

chart = sns.catplot(
    x='Country_Year',
    y='Prediction',
    data=all_resultsm,
    scale=0.2,
    kind='point',
    height=4,
    aspect=3,
    hue='Model')

chart.set(xlabel='Country and Year', ylabel='Slavery (% pop)')
chart.set_xticklabels(rotation=90, horizontalalignment='right', size=7)
chart._legend.set_title("Prediction")
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.savefig(save_path+'Rash_model_predictions.pdf')


# ~Plot clearer:
fig, axs = plt.subplots()
plt.figure(figsize=(20,7))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})

g = sns.lineplot(x='Country_Year', y='Rashomon 1', data=all_results, linewidth=2,
                color="plum", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Rashomon 2', data=all_results, linewidth=2,
                color="mediumturquoise", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Rashomon 3', data=all_results, linewidth=2,
                color="grey", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Rashomon 4', data=all_results, linewidth=2,
                color="tan", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Rashomon 5', data=all_results, linewidth=2,
                color="lightgrey", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Actual', data=all_results, linewidth=4,
                color="dodgerblue", style=True)

sns.lineplot(x='Country_Year', y='Best Model', data=all_results, linewidth=4,
                color="orange")

custom_lines = [Line2D([0], [0], color="dodgerblue", lw=4),
                Line2D([0], [0], color="orange", lw=4),
                Line2D([0], [0], color="plum", lw=2, ls='--'),
                Line2D([0], [0], color="mediumturquoise", lw=2, ls='--'),
                Line2D([0], [0], color="grey", lw=2, ls='--'),
                Line2D([0], [0], color="tan", lw=2, ls='--'),
                Line2D([0], [0], color="lightgrey", lw=2, ls='--')
                ]

g.set(xlabel=' ', ylabel='Slavery (% pop)')
g.set_xticklabels(all_results["Country_Year"], rotation=90, horizontalalignment='center', size=12)
# plt.legend([],[], frameon=False)
g.legend(handles=custom_lines, loc='upper right', title="Prevalence Estimate", frameon=False, markerscale=2,
           labels=["Actual", "Best Model", "Rashomon 1", "Rashomon 2", "Rashomon 3", "Rashomon 4", "Rashomon 5"])
axs.set(ylim=(0,2.5))
plt.margins(x=0.005)
plt.subplots_adjust(bottom=0.1)
plt.tight_layout()
plt.savefig(save_path+"NEW_Rash_model_predictions.pdf")
plt.close()


# # plot line - ordered by error

all_results["Error"] = abs(all_results["Actual"]-all_results["Best Model"])
all_results.sort_values(by="Error", inplace=True, axis=0, ascending=True)

fig, axs = plt.subplots()
plt.figure(figsize=(20,7))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})

g = sns.lineplot(x='Country_Year', y='Rashomon 1', data=all_results, linewidth=2,
                color="plum", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Rashomon 2', data=all_results, linewidth=2,
                color="mediumturquoise", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Rashomon 3', data=all_results, linewidth=2,
                color="grey", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Rashomon 4', data=all_results, linewidth=2,
                color="tan", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Rashomon 5', data=all_results, linewidth=2,
                color="lightgrey", style=True, dashes=[(3,1)])

sns.lineplot(x='Country_Year', y='Actual', data=all_results, linewidth=4,
                color="dodgerblue", style=True)

sns.lineplot(x='Country_Year', y='Best Model', data=all_results, linewidth=4,
                color="orange")

custom_lines = [Line2D([0], [0], color="dodgerblue", lw=4),
                Line2D([0], [0], color="orange", lw=4),
                Line2D([0], [0], color="plum", lw=2, ls='--'),
                Line2D([0], [0], color="mediumturquoise", lw=2, ls='--'),
                Line2D([0], [0], color="grey", lw=2, ls='--'),
                Line2D([0], [0], color="tan", lw=2, ls='--'),
                Line2D([0], [0], color="lightgrey", lw=2, ls='--')
                ]

g.set(xlabel=' ', ylabel='Slavery (% pop)')
g.set_xticklabels(all_results["Country_Year"], rotation=90, horizontalalignment='center', size=12)
# plt.legend([],[], frameon=False)
g.legend(handles=custom_lines, loc='upper left', title="Prevalence Estimate", frameon=False, markerscale=2,
           labels=["Actual", "Best Model", "Rashomon 1", "Rashomon 2", "Rashomon 3", "Rashomon 4", "Rashomon 5"])
axs.set(ylim=(0,2.5))
plt.margins(x=0.005)
plt.subplots_adjust(bottom=0.1)
plt.tight_layout()
plt.savefig(save_path+"NEW_Rash_model_predictions_ORDERED_ERR_ASC.pdf")
plt.close()