from Functions.Preprocessing.Normalise import Normalise
from matplotlib.lines import Line2D
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.decomposition import NMF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np

print('Running')

# sample data : ----------------------------------------------------------------------------------
data_path = "Data/"
sdf = pd.read_csv(data_path + "Data.csv", index_col=False)

# Create index's:
Index = sdf[['Country', 'Data_year']]
Index_list = []
for i in range(0, 70):
    Index_list.append(str(Index.Country[i]) + '_' + str(Index.Data_year[i]))
    Index_list[i] = Index_list[i].replace(" ", "_")

# Create X and y:
SX = sdf.drop(["SLAVERY", "Country", "Data_year", "Region"], axis=1)
SX = Normalise(SX)
SX.index = Index_list
SX.rename(index= {'Democratic_Republic_of_the_Congo_2018': 'Dem_Rep_Congo_2018'}, inplace=True)

Sy = sdf['SLAVERY']
Sy = pd.DataFrame(Sy)
Sy.index = Index_list
Sy.rename(index= {'Democratic_Republic_of_the_Congo_2018': 'Dem_Rep_Congo_2018'}, inplace=True)

sdf = pd.concat([SX, Sy], axis=1)
# out of sample data : ----------------------------------------------------------------------------

oosdf = pd.read_csv("Data/Out_of_Sample_Data/OOS_Data.csv", index_col=False)

OS_X = oosdf.drop(['SLAVERY','Unnamed: 0','Country','Data_year'], axis=1)
OS_y = oosdf['SLAVERY']

# Create index:
Index = oosdf[['Country', 'Data_year']]
Index_list = []
for i in range(0, oosdf.shape[0]):
    Index_list.append(str(Index.Country[i]) + '_' + str(Index.Data_year[i]))
    Index_list[i] = Index_list[i].replace(" ", "_")

not_in = list(oosdf.columns[~oosdf.columns.isin(sdf)])

OS_X.index = Index_list
OS_y.index = Index_list

OS_df = pd.concat([OS_X, OS_y], axis=1)

# join sample and out of sample data:

concat_all = pd.concat([sdf, OS_df], axis=0, keys= ['s','os'], join='outer', ignore_index=False,
          levels=None, names=None, verify_integrity=True, copy=True)

new_rows = OS_df[~OS_df.index.isin(sdf.index)]

concat = pd.concat([sdf,new_rows], axis=0,join='outer', ignore_index=False,
          levels=None, names=None, verify_integrity=True, copy=True)

# Combined, new X and y:
X = concat.drop('SLAVERY', axis=1)
X = Normalise(X)
y = concat['SLAVERY']

new_r = y[~y.index.isin(OS_y.index)]

# best model params -------------------------------------------------------------:

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

#  check the LOO MAE:
cv = LeaveOneOut()
m1_score = cross_val_score(pipe, SX, Sy, cv=cv, scoring="neg_mean_absolute_error")
mae_m1 = abs(m1_score).mean()
print(mae_m1)

# get test (hold out) and train (GSI_70)

test_X = X[~X.index.isin(SX.index)]
test_y = y[~y.index.isin(Sy.index)]
test_y = pd.DataFrame(test_y, index= test_y.index)

# make predictions

pipe.fit(SX, Sy)
y_pred_holdout = pipe.predict(test_X)
y_pred_holdout = pd.DataFrame(y_pred_holdout, index=test_y.index)
y_pred_holdout["Country_Year"] = y_pred_holdout.index
#===================================================================================
#  Bootstrappping
#===================================================================================
# distribution of errors (LOO prediction on training data)
save_path ="Outputs/Plot_Model_Predictions/Out_of_Sample_Predictions/"
n = 10000

# in a loop:
#
# indexes = []
# sample_actual = []
# sample_scores = []
# sample_predictions = []
# for i in range(0,n):
#     np.random.seed(i)
#     x_array = SX.to_records(index=True)
#     y_array = Sy.to_records(index=True)
#     rand_index = np.random.choice(a=x_array.shape[0], size=x_array.shape[0], replace=True)
#     resample_x = pd.DataFrame(x_array[rand_index])
#     resample_y = pd.DataFrame(y_array[rand_index])
#     resample_x.set_index('index', drop=True, inplace=True)
#     resample_y.set_index('index', drop=True, inplace=True)
#
#     # fit model:
#     m1 = pipe.fit(resample_x, resample_y)
#
#     # loo predict:
#     y_pred_holdout = m1.predict(test_X)
#     y_pred_holdout = pd.DataFrame(y_pred_holdout, index=test_y.index)
#
#     sample_pred_df = pd.DataFrame(y_pred_holdout)
#     sample_pred_df.index = test_y.index
#
#     # Save to list:
#     sample_predictions.append(sample_pred_df)
#
# # join all datasets together:
# all_predictions = pd.concat(sample_predictions, axis=0, join='outer')
# all_predictions["Country_Year"]=all_predictions.index
# all_predictions.columns = ["OOS Prediction","Country_Year"]
# all_predictions.sort_values(by = "Country_Year", inplace=True)
# all_predictions.to_csv(save_path+"Bootstrapped_Predictions_n{}.csv".format(n), index=False)
#
# ------------------------------------------------------------------------------
# df = pd.read_csv(save_path+"Bootstrapped_Predictions_n{}.csv".format(n))
#
# actual = test_y
# actual["Country_Year"] = actual.index
#
# results = actual.merge(y_pred_holdout, how='left', on='Country_Year')
# results = df.merge(results, how='left', on='Country_Year')
# results.columns = ["BS Prediction", "Country_Year", "Actual", "Model Prediction"]
#
# #  get 2018 only:
# year_list = []
# country_list = []
# for i in range(0,results.shape[0]):
#     year_list.append(str(results.Country_Year[i])[-4:])
#     country_list.append(str(results.Country_Year[i])[:-5])
#     country_list[i] = country_list[i].replace("_", " ")
#
# results['Year'] = year_list
# results['Country'] = country_list
# results.drop('Country_Year', axis=1, inplace=True)
# results2018=results[results['Year']=='2018']
# results2018.to_csv(save_path+"Bootstrapped_Predictions_n{}_2018.csv".format(n))

# ======================================================================
# and where <10% missing data:
results2018 = pd.read_csv(save_path+"Bootstrapped_Predictions_n{}_2018.csv".format(n))
missing_data = pd.read_csv("Data/Meta_Data/Missing_data.csv")
join2018 = missing_data[missing_data['Data_year']==2018]
join2018 = join2018[["Country","Missing_perc"]]
merge = results2018.merge(join2018, on='Country', how='left')
merge = merge[["Country","Missing_perc","Model Prediction","Actual", "BS Prediction"]]
merge.columns =["Country","Missing_perc","Model","GSI", "BS Prediction"]

# Get summary table:
summ = merge[["Country","Missing_perc","Model","GSI"]]
summ.drop_duplicates(inplace=True)
summ["Model"]=round(summ["Model"],2)
summ.to_csv("Outputs/Plot_Model_Predictions/Out_of_Sample_Predictions/Summary_tab.csv")


miss_perc = 10
merge_10 = merge[merge.Missing_perc < miss_perc]
merge_10.columns=["Country","Missing Data (%)","Model","GSI","BS Prediction"]

# plot : -----------------------------------------------------------------------------------------

# plot line - ordered by error
merge_10["Diff"] = abs(merge_10["GSI"]-merge_10["Model"])

merge_10.sort_values(by='Diff', inplace=True, axis=0, ascending=True)

fig, axs = plt.subplots()
plt.figure(figsize=(20,7))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})
g = sns.boxplot(x='Country', y='BS Prediction', data=merge_10, color="whitesmoke", saturation=0.6,
                showfliers=False, linewidth=0.6)

sns.lineplot(x='Country', y='GSI', data=merge_10, linewidth=15,
                size=10, color="mediumseagreen")

sns.lineplot(x='Country', y='Model', data=merge_10, linewidth=15,
                size=10, color="orange")

custom_lines = [Line2D([0], [0], color="orange", lw=4),
                Line2D([0], [0], color="mediumseagreen", lw=4)]

g.set(xlabel=' ', ylabel='Slavery (% pop)')
g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment='center', size=12)
# plt.legend([],[], frameon=False)
g.legend(handles=custom_lines, loc='upper left', title="Prevalence Estimate", frameon=False, markerscale=2,
           labels=["Our Best Model","GSI Model"])
axs.set(ylim=(0,2.5))
plt.subplots_adjust(bottom=0.1)
plt.tight_layout()
plt.savefig(save_path+'NEW090221Plot_OSS_Predictions_boxplot_n{}_ORDER_DIFF_ASC.pdf'.format(n))
plt.close()

# # plot with missing data:
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# plt.figure(figsize=(20,7))
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})
# sns.boxplot(x='Country', y='BS Prediction', data=merge_10, color="whitesmoke", saturation=0.6,
#                 showfliers=False, linewidth=0.6, ax=ax1)
#
# sns.lineplot(x='Country', y='GSI', data=merge_10, linewidth=15,
#                 size=10, color="mediumseagreen", ax=ax1)
#
# sns.lineplot(x='Country', y='Model', data=merge_10, linewidth=15,
#                 size=10, color="orange", ax=ax1)
#
# sns.lineplot(x='Country', y='Missing Data (%)', data=merge_10, linewidth=15,
#                 size=5, color="lightgrey", ax=ax2)
#
# custom_lines = [Line2D([0], [0], color="orange", lw=4),
#                 Line2D([0], [0], color="mediumseagreen", lw=4)]
#
# ax1.set(xlabel=' ', ylabel='Slavery (% pop)')
# ax1.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment='center', size=12)
# # plt.legend([],[], frameon=False)
# ax1.legend(handles=custom_lines, loc='upper left', title="Prevalence Estimate", frameon=False, markerscale=2,
#            labels=["Best Model","GSI Model"])
# ax1.set(ylim=(0,2.5))
# ax2.set_ylabel("% Missing IV Data")
# plt.subplots_adjust(bottom=0.1)
# plt.tight_layout()
# plt.show()
# plt.savefig(save_path+'NEWPlot_OSS_Predictions_boxplot_n{}_ORDER_DIFF_ASC_MISSINGDATA.pdf'.format(n))
# plt.close()