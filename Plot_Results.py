import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

open_dict = {"Results/All_features/":
                 {"NMF_LM": "NMF->LM",
                  "NMF_DT": "NMF->DT",
                  "NMF_DT_MF": "NMF->DT (MF)",
                  "NMF_RF": "NMF->RF",
                  "PLS": "PLS",
                  "LASSO": "Lasso",
                  "DT": "DT",
                  "DT_MF": "DT (MF)",
                  "RF": "RF"
                  },
             "Results/Theory_features/":
                 {"NMF_LM": "NMF->LM",
                  "NMF_DT": "NMF->DT",
                  "NMF_DT_MF": "NMF->DT (MF)",
                  "NMF_RF": "NMF->RF",
                  "PLS": "PLS",
                  "LASSO": "Lasso",
                  "DT": "DT",
                  "DT_MF": "DT (MF)",
                  "RF": "RF"
                  }}

open_dict_long_names = {"/Results/All_features/":
                            {"NMF_LM": "NMF->Linear Regression",
                             "NMF_DT": "NMF->DT (max_features=n_features)",
                             "NMF_DT_MF": "NMF->DT (max features as param)",
                             "NMF_RF": "NMF->Random Forest",
                             "PLS": "Partial Least Squares",
                             "LASSO": "Lasso regression",
                             "DT": "DT (max_features=n_features)",
                             "DT_MF": "DT (max features as param)",
                             "RF": "Random Forest"
                             },
                        "/Results/Theory_features/":
                            {"NMF_LM": "NMF->Linear Regression",
                             "NMF_DT": "NMF->DT (max_features=n_features)",
                             "NMF_DT_MF": "NMF->DT (max features as param)",
                             "NMF_RF": "NMF->Random Forest",
                             "PLS": "Partial Least Squares",
                             "LASSO": "Lasso regression",
                             "DT": "DT (max_features=n_features)",
                             "DT_MF": "DT (max features as param)",
                             "RF": "Random Forest"
                             }}

dict = {'N_features': [], 'Model_name': [], 'Error': []}

error_dict = {'RMSE': ['mean_test_neg_root_mean_squared_error', 'rank_test_neg_root_mean_squared_error'],
              'MAE': ['mean_test_neg_mean_absolute_error', 'rank_test_neg_mean_absolute_error'],
              'Variance_explained': ['mean_test_explained_variance', 'rank_test_explained_variance']}

# Get new data -------------------------------------------------------------------------
home_dir = "/Users/rlavelle-hill/OneDrive - The Alan Turing Institute/Documents/ml-slavery/"
file_end = "_Results.txt"
for folder in open_dict:
    for model_type, model_name in open_dict[folder].items():
        file_string=str(folder)+str(model_type)+str(file_end)
        f = open(file_string)
        lines = f.readlines()
        error = str(lines[1]).split(':',1)[1]
        error=round(float(error),3)
        n_features = folder.split("/",2)[1]

        dict['N_features'].append(n_features)
        dict['Model_name'].append(model_name)
        dict['Error'].append(error)

results_df = pd.DataFrame.from_dict(dict)
results_df.to_csv("Results/All_Results.csv")


# Plot Summary Results -------------------------------------------------------------------------

fig_save_path = 'Outputs/Plot_All_Results/'
dd = pd.melt(results_df, id_vars=['N_features', 'Model_name'])
plt.figure(figsize=(20, 8))
ymin = round(min(dd.value) - 0.05, 1)
ymax = round(max(dd.value) + 0.05, 1)

chart = sns.catplot(
        data=dd,
        x='Model_name',
        y='value',
        kind='bar',
        palette='Set2',
        col='N_features',
        aspect=1
    )
chart.set_xticklabels(rotation=65, horizontalalignment='right')
chart.set(xlabel="Model", ylabel="MAE")
# (chart.set_axis_labels("Model", key)
#      .set_titles("{col_name}")
#      .set(ylim=(ymin, ymax)))

plt.subplots_adjust(bottom=0.3, left=0.1)

plt.savefig(str(fig_save_path) + "All_Results.png")

print('done')


