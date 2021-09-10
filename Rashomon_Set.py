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
# collate all rashomon results

results_all = []
results = pd.DataFrame()
home_dir = "/Users/rlavelle-hill/OneDrive - The Alan Turing Institute/Documents/ml-slavery/"
file_end = "_Rash_results.csv"
for folder in open_dict:
    for model_type, model_name in open_dict[folder].items():
        file_string=str(folder)+str(model_type)+str(file_end)
        df = pd.read_csv(file_string)
        # df.rename(columns={"Unnamed: 0": "MAE"}, errors="raise", inplace=True)
        dfn = pd.DataFrame(df.iloc[:,0])
        dfn.columns = ['MAE']
        dfn['Model'] = model_name
        dfn['Features'] = folder.split('/', 2)[1]
        results = results.append(dfn)
        dfn = dfn.merge(df, left_index=True, right_index=True)
        results_all.append(dfn)

results_all = pd.concat(results_all)
results_all.drop('Unnamed: 0', axis=1, inplace=True)

# get rid of > rash_thresh
# rash_thresh = 0.2340852835055713
rash_thresh = 0.27
save_path = "Outputs/Rashomon_Set/"
results=results[results.MAE < rash_thresh]
results_all=results_all[results_all.MAE < rash_thresh]
results.sort_values(by='MAE', axis=0, inplace=True)
results_all.sort_values(by='MAE', axis=0, inplace=True)
results.to_csv(save_path+'Rash_Results_all_{}.csv'.format(rash_thresh))
results_all.to_csv(save_path+'Rash_Results_all_parms_{}.csv'.format(rash_thresh), index=False)

# plot MAE by model
results["Model Type"] = results["Features"] + " : " + results["Model"]
results["Model Type"] = results["Model Type"].str.replace("_"," ", regex=False)

plt.figure(figsize=(15, 8))
sns.set_theme(style="whitegrid")
g = sns.catplot(x="Model Type", y="MAE", hue="Model Type",
                data=results, palette='Set2')
g.set_xticklabels(rotation=90)
plt.tight_layout()
g.savefig(save_path+"NEW_MAE_all_{}.png".format(rash_thresh))

# Variable Importance Calculation
print('done!')