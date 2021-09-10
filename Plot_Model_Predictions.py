import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Functions.Preprocessing.Normalise import Normalise
from matplotlib.lines import Line2D
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.decomposition import NMF


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
#  -----------------------------------------------------------------------------
save_path = "Outputs/Plot_Model_Predictions/"

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

n = 10000

# indexes = []
# sample_actual = []
# sample_scores = []
# sample_predictions = []
# for i in range(0,n):
#     np.random.seed(i)
#     x_array = X.to_records(index=True)
#     y_array = y.to_records(index=True)
#     rand_index = np.random.choice(a=x_array.shape[0], size=x_array.shape[0], replace=True)
#     resample_x = pd.DataFrame(x_array[rand_index])
#     resample_y = pd.DataFrame(y_array[rand_index])
#     resample_x.set_index('index', drop=True, inplace=True)
#     resample_y.set_index('index', drop=True, inplace=True)
#
#     # # fit model:
#     pipe.fit(resample_x, resample_y)
#
#     # loo predict:
#     sample_score = round(abs(cross_val_score(pipe, resample_x, resample_y, cv=70, scoring="neg_mean_absolute_error").mean()),3)
#     sample_predict = cross_val_predict(pipe, resample_x, resample_y, cv=70)
#
#     sample_pred_df = pd.DataFrame(sample_predict)
#     sample_pred_df.index = resample_y.index
#
#     # Save to lists:
#     indexes.append(resample_y.index)
#     sample_actual.append(resample_y)
#     sample_scores.append(sample_score)
#     sample_predictions.append(sample_pred_df)
#
# # join all datasets together:
# all_predictions = pd.concat(sample_predictions, axis=0, join='outer')
# all_predictions.to_csv(save_path+"Bootstrapped_Predictions_n{}.csv".format(n))

# ----------------------------------------------------------------------------------------------------------
# load data and plot
df = pd.read_csv(save_path+"Bootstrapped_Predictions_n{}.csv".format(n))
df.set_index('index', inplace=True)
df.rename(index= {'Democratic_Republic_of_the_Congo_2018': 'Dem_Rep_Congo_2018'}, inplace=True)
df.reset_index(inplace=True)
df.columns=["Country_year","Prediction"]
df.sort_values(by="Country_year", inplace=True)

actual = y
actual["Country_year"] = actual.index

cv = LeaveOneOut()
ypred = cross_val_predict(pipe, X, y.SLAVERY, cv=cv)
actual["Our Model"] = ypred

results = df.merge(actual, how='left', on='Country_year')
results.columns = ["Country_year","Prediction", "Actual", "Our Model"]


# plot line
fig, axs = plt.subplots()
plt.figure(figsize=(20,7))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})
g = sns.boxplot(x='Country_year', y='Prediction', data=results, color="whitesmoke", saturation=0.6,
                showfliers=False, linewidth=0.6)

sns.lineplot(x='Country_year', y='Actual', data=results, linewidth=15,
                size=10, color="dodgerblue")

sns.lineplot(x='Country_year', y='Our Model', data=results, linewidth=15,
                size=10, color="orange")

custom_lines = [Line2D([0], [0], color="dodgerblue", lw=4),
                Line2D([0], [0], color="orange", lw=4)]

g.set(xlabel=' ', ylabel='Slavery (% pop)')
g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment='center', size=12)
g.legend(handles=custom_lines, loc='upper right', title="Prevalence Estimate", frameon=False, markerscale=2,
           labels=["Actual", "LOOCV Prediction"])
axs.set(ylim=(0,2.5))
plt.subplots_adjust(bottom=0.1)
plt.tight_layout()
plt.savefig(save_path+'Plot_training_boxplot_n{}.png'.format(n))
plt.close()

# plot line - ordered by error

results["Error"] = abs(results["Actual"]-results["Our Model"])
results.sort_values(by="Error", inplace=True, axis=0, ascending=True)

fig, axs = plt.subplots()
plt.figure(figsize=(20,7))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})
g = sns.boxplot(x='Country_year', y='Prediction', data=results, color="whitesmoke", saturation=0.6,
                showfliers=False, linewidth=0.6)

sns.lineplot(x='Country_year', y='Actual', data=results, linewidth=15,
                size=10, color="dodgerblue")

sns.lineplot(x='Country_year', y='Our Model', data=results, linewidth=15,
                size=10, color="orange")

custom_lines = [Line2D([0], [0], color="dodgerblue", lw=4),
                Line2D([0], [0], color="orange", lw=4)]

g.set(xlabel=' ', ylabel='Slavery (% pop)')
g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment='center', size=12)
g.legend(handles=custom_lines, loc='upper left', title="Prevalence Estimate", frameon=False, markerscale=2,
           labels=["Actual", "LOOCV Prediction"])
axs.set(ylim=(0,2.5))
plt.subplots_adjust(bottom=0.1)
plt.tight_layout()
plt.savefig(save_path+'Plot_training_boxplot_n{}_ORDERED_ERROR_ASC.png'.format(n))
plt.close()

# plot dot ------------=
fig, axs = plt.subplots()
plt.figure(figsize=(20,7))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
g = sns.boxplot(x='Country_year', y='Prediction', data=results, color="whitesmoke", saturation=0.6,
                showfliers=False, linewidth=0.6)

sns.pointplot(x='Country_year', y='Actual', data=results, markers='_',
                size=2, color="dodgerblue",join=False)

sns.pointplot(x='Country_year', y='Our Model', data=results, markers='_',
                size=2, color="orange", join=False)

custom_lines = [Line2D([0], [0], marker='_', color="dodgerblue", markersize=2),
                Line2D([0], [0], marker='_', color="orange", markersize=2)]

g.set(xlabel=' ', ylabel='Slavery (% pop)')
g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment='center', size=12)
g.legend(handles=custom_lines, loc='upper right', title="Prevalence Estimate", frameon=False, markerscale=2,
           labels=["Actual", "LOOCV Prediction"])
axs.set(ylim=(0,2.5))
plt.subplots_adjust(bottom=0.1)
plt.tight_layout()
plt.savefig(save_path+'Plot_training_boxplot_n{}_scatter.png'.format(n))
plt.close()

# plot dot SORTED ERROR------------

results["Error"] = abs(results["Actual"]-results["Our Model"])
results.sort_values(by="Error", inplace=True, axis=0, ascending=True)

fig, axs = plt.subplots()
plt.figure(figsize=(20,7))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
g = sns.boxplot(x='Country_year', y='Prediction', data=results, color="whitesmoke", saturation=0.6, whis=0.95,
                showfliers=False, linewidth=0.6)

sns.pointplot(x='Country_year', y='Actual', data=results, markers='_',
                size=2, color="dodgerblue",join=False)

sns.pointplot(x='Country_year', y='Our Model', data=results, markers='_',
                size=2, color="orange", join=False)

custom_lines = [Line2D([0], [0], marker='_', color="dodgerblue", markersize=2),
                Line2D([0], [0], marker='_', color="orange", markersize=2)]

g.set(xlabel=' ', ylabel='Slavery (% pop)')
g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment='center', size=12)
g.legend(handles=custom_lines, loc='upper left', title="Prevalence Estimate", frameon=False, markerscale=2,
           labels=["Actual", "LOOCV Prediction"])
axs.set(ylim=(0,2.5))
plt.subplots_adjust(bottom=0.1)
plt.tight_layout()
plt.savefig(save_path+'Plot_training_boxplot_n{}_scatterORDERED_ERROR_ASC_95.png'.format(n))
plt.close()

# plot dot SORTED ERROR------------

results["Error"] = abs(results["Actual"]-results["Our Model"])
results.sort_values(by="Actual", inplace=True, axis=0, ascending=False)

fig, axs = plt.subplots()
plt.figure(figsize=(20,7))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
g = sns.boxplot(x='Country_year', y='Prediction', data=results, color="whitesmoke", saturation=0.6,
                showfliers=False, linewidth=0.6)

sns.pointplot(x='Country_year', y='Actual', data=results, markers='_',
                size=2, color="dodgerblue",join=False)

sns.pointplot(x='Country_year', y='Our Model', data=results, markers='_',
                size=2, color="orange", join=False)

custom_lines = [Line2D([0], [0], marker='_', color="dodgerblue", markersize=2),
                Line2D([0], [0], marker='_', color="orange", markersize=2)]

g.set(xlabel=' ', ylabel='Slavery (% pop)')
g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment='center', size=12)
g.legend(handles=custom_lines, loc='upper right', title="Prevalence Estimate", frameon=False, markerscale=2,
           labels=["Actual", "LOOCV Prediction"])
axs.set(ylim=(0,2.5))
plt.subplots_adjust(bottom=0.1)
plt.tight_layout()
plt.savefig(save_path+'Plot_training_boxplot_n{}_scatterORDERED_ACTUAL_DESC.png'.format(n))
plt.close()


# ADD MISSING DATA to plot line - ordered by error
miss = pd.read_csv("Data/Meta_Data/Missing_data.csv")

Country_Year_List = []
for i in range(0,miss.shape[0]):
    Country = miss['Country'][i].replace(" ","_")
    Country_Year = Country + '_' + str(miss['Data_year'][i])
    Country_Year_List.append(Country_Year)

miss["Country_year"] =Country_Year_List
results = results.merge(miss, on=["Country_year"], how='left')

results["Error"] = abs(results["Actual"]-results["Our Model"])
results.sort_values(by="Error", inplace=True, axis=0, ascending=True)
#
#
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# plt.figure(figsize=(20,7))
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})
# sns.boxplot(x='Country_year', y='Prediction', data=results, color="whitesmoke", saturation=0.6,
#                 showfliers=False, linewidth=0.6, ax=ax1)
#
# sns.lineplot(x='Country_year', y='Actual', data=results, linewidth=15,
#                 size=10, color="dodgerblue", ax=ax1)
#
# sns.lineplot(x='Country_year', y='Our Model', data=results, linewidth=15,
#                 size=10, color="orange", ax=ax1)
#
# sns.lineplot(x='Country_year', y='Missing_perc', data=results, linewidth=15,
#                 size=5, color="lightgrey", ax=ax2)
#
# custom_lines = [Line2D([0], [0], color="dodgerblue", lw=4),
#                 Line2D([0], [0], color="orange", lw=4)]
#
# ax1.set(xlabel=' ', ylabel='Slavery (% pop)')
# ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, horizontalalignment='center', size=12)
# ax1.legend(handles=custom_lines, loc='upper left', title="Prevalence Estimate", frameon=False, markerscale=2,
#            labels=["Actual", "LOOCV Prediction"])
# ax1.set(ylim=(0,2.5))
# ax2.set_ylabel("% Missing IV Data")
# plt.subplots_adjust(bottom=0.1)
# plt.tight_layout()
# plt.show()
# plt.savefig(save_path+'Plot_training_boxplot_n{}_missing_perc.png'.format(n))
# plt.close()


print('done!')