import pandas as pd
from Functions.Interpretation.Plot_NMF import Get_W_matrix, Get_H_matrix
from Functions.Preprocessing.Normalise import Normalise
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from Functions.Interpretation.Rashomon_Model_Feature_Importance import NMF_DT_MF, NMF_DT, NMF_RF, DT, DT_MF, RF
# =========================================================
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

# =========================================================
open_dict = {"Results/All_features/":
                 {"NMF_LM": "NMF->LM",
                  "NMF_DT": "NMF->DT",
                  "NMF_DT_MF": "NMF->DT (MF)",
                  "NMF_RF": "NMF->RF",
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
                             "LASSO": "Lasso regression",
                             "DT": "DT (max_features=n_features)",
                             "DT_MF": "DT (max features as param)",
                             "RF": "Random Forest"
                             }}

keep_vars = ['KOF_Globalis','Work_rightCIRI','Trade_open','GDPpc',
 'Armedcon','Poverty', 'Stunting_u5s','Undernourish','Wasting_u5s','Maternal_mort',
 'Neonatal_mort','Literacy_15_24yrs','F_school','ATMs','Child_lab','Unemploy','Infrastruct',
 'Internet_use','Broadband','Climate_chg_vuln','CPI','Minority_rule','Freemv_M',
 'Freemv_F','Free_discuss','Soc_powerdist','Democ',
 'Sexwrk_condom','Sexwrk_Syphilis','AIDS_Orph','Rape_report',
 'Rape_enclave','Phys_secF','Gender_equal']

home_dir = "/Users/rlavelle-hill/OneDrive - The Alan Turing Institute/Documents/ml-slavery/"
file_end = "_Rash_results.csv"

rashomon_threshold = 0.2340852835055713
tol=0.005
max_iter=250
n_repeats = 5
d_nmf_all = {}
d_nmf_theory = {}
d_non_nmf_all = {}
d_non_nmf_theory = {}
d_model = {}
all_results = {}
for folder in open_dict:
    for model_type, model_name in open_dict[folder].items():
        file_string=str(folder)+str(model_type)+str(file_end)
        features = folder.split('/', 2)[1]
        df = pd.read_csv(file_string)
        df.rename(columns={'Unnamed: 0':'MAE'}, inplace=True)
        df = df[df.MAE<rashomon_threshold]
        if df.shape[0] > 0:
            if 'NMF->' in model_name:
                if 'DT' in model_name:
                    if '(MF)' in model_name:
                        for index, row in df.iterrows():
                            print("Evaluating "+ features + " " + model_name+", MAE :" +str(row['MAE']))
                            if features == "All_features":
                                all_results[row['MAE']]=NMF_DT_MF(X=X, y=y, row=row, max_iter=max_iter, tol=tol, n_repeats=n_repeats)
                            if features == "Theory_features":
                                X = X[keep_vars]
                                all_results[row['MAE']]=NMF_DT_MF(X=X, y=y, row=row, max_iter=max_iter, tol=tol, n_repeats=n_repeats)
                    else:
                        for index, row in df.iterrows():
                            print("Evaluating " + features + " " + model_name + ", MAE :" + str(row['MAE']))
                            if features == "All_features":
                                all_results[row['MAE']] = NMF_DT(X=X, y=y, row=row, max_iter=max_iter, tol=tol,
                                                                    n_repeats=n_repeats)
                            if features == "Theory_features":
                                X = X[keep_vars]
                                all_results[row['MAE']] = NMF_DT(X=X, y=y, row=row, max_iter=max_iter, tol=tol, n_repeats=n_repeats)
                else:
                    for index, row in df.iterrows():
                        print("Evaluating " + features + " " + model_name + ", MAE :" + str(row['MAE']))
                        if features == "All_features":
                            all_results[row['MAE']] = NMF_RF(X=X, y=y, row=row, max_iter=max_iter, tol=tol,
                                                                n_repeats=n_repeats)
                        if features == "Theory_features":
                            X = X[keep_vars]
                            all_results[row['MAE']] = NMF_RF(X=X, y=y, row=row, max_iter=max_iter, tol=tol,
                                                                n_repeats=n_repeats)
            else:
                if 'DT' in model_name:
                    if '(MF)' in model_name:
                        for index, row in df.iterrows():
                            print("Evaluating "+ features + " " + model_name+", MAE :" +str(row['MAE']))
                            if features == "All_features":
                                all_results[row['MAE']] = DT_MF(X=X, y=y, row=row, n_repeats=n_repeats)
                            if features == "Theory_features":
                                X = X[keep_vars]
                                all_results[row['MAE']] = DT_MF(X=X, y=y, row=row, n_repeats=n_repeats)
                    else:
                        for index, row in df.iterrows():
                            print("Evaluating "+ features + " " + model_name+", MAE :" +str(row['MAE']))
                            if features == "All_features":
                                all_results[row['MAE']] = DT(X=X, y=y, row=row, n_repeats=n_repeats)
                            if features == "Theory_features":
                                X = X[keep_vars]
                                all_results[row['MAE']] = DT(X=X, y=y, row=row, n_repeats=n_repeats)
                else:
                    for index, row in df.iterrows():
                        print("Evaluating "+ features + " " + model_name+", MAE :" +str(row['MAE']))
                        if features == "All_features":
                            all_results[row['MAE']] = RF(X=X, y=y, row=row, n_repeats=n_repeats)
                        if features == "Theory_features":
                            X = X[keep_vars]
                            all_results[row['MAE']] = RF(X=X, y=y, row=row, n_repeats=n_repeats)

save_path = "Outputs/Rashomon_Set/Model_Variable_Importance/"
results = pd.DataFrame.from_dict(all_results)
results.to_csv(save_path+"Model_Variable_Importance_Threshold_{}.csv".format(rashomon_threshold))


print("done!")