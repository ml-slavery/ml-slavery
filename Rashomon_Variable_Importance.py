import pandas as pd
from Functions.Interpretation.Plot_NMF import Get_W_matrix, Get_H_matrix
from Functions.Preprocessing.Normalise import Normalise
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
                                H = Get_H_matrix(X=X, best_alpha=row['nmf__alpha'],
                                                            best_n=row['nmf__n_components'],
                                                            best_solver=row['nmf__solver'],
                                                            random_state=row['dt__random_state'],
                                                            max_iter=max_iter, tol=tol)
                                W = Get_W_matrix(X=X, best_alpha=row['nmf__alpha'],
                                                            best_n=row['nmf__n_components'],
                                                            best_solver=row['nmf__solver'],
                                                            random_state=row['dt__random_state'],
                                                            max_iter=max_iter, tol=tol)
                                nmf_dt_mf = DecisionTreeRegressor(
                                    random_state=row['dt__random_state'],
                                    min_samples_split=row['dt__min_samples_split'],
                                    max_features=row['dt__max_features'],
                                    max_depth=row['dt__max_depth'])

                                nmf_dt_mf.fit(W,y)
                                r = permutation_importance(nmf_dt_mf, W, y, n_repeats=n_repeats, random_state=0,
                                                           scoring="neg_mean_absolute_error")
                                imp = pd.DataFrame(r.importances_mean)
                                comp = pd.DataFrame(W.columns.values)
                                perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                                perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                                # Normalise the loadings for each component to be between 0 and 1:
                                H = Normalise(H.transpose())

                                for index, row in H.iterrows():
                                     vimp = row*perm_imp['Permutation_Importance']
                                     if index in d_nmf_all:
                                         d_nmf_all[index].append(vimp.sum())

                                     else:
                                         d_nmf_all[index]=[vimp.sum()]

                            if features == "Theory_features":
                                X = X[keep_vars]
                                H = Get_H_matrix(X=X, best_alpha=row['nmf__alpha'],
                                                 best_n=row['nmf__n_components'],
                                                 best_solver=row['nmf__solver'],
                                                 random_state=row['dt__random_state'],
                                                 max_iter=max_iter, tol=tol)
                                W = Get_W_matrix(X=X, best_alpha=row['nmf__alpha'],
                                                 best_n=row['nmf__n_components'],
                                                 best_solver=row['nmf__solver'],
                                                 random_state=row['dt__random_state'],
                                                 max_iter=max_iter, tol=tol)
                                nmf_dt_mf = DecisionTreeRegressor(
                                    random_state=row['dt__random_state'],
                                    min_samples_split=row['dt__min_samples_split'],
                                    max_features=row['dt__max_features'],
                                    max_depth=row['dt__max_depth'])

                                nmf_dt_mf.fit(W, y)
                                r = permutation_importance(nmf_dt_mf, W, y, n_repeats=n_repeats, random_state=0,
                                                           scoring="neg_mean_absolute_error")
                                imp = pd.DataFrame(r.importances_mean)
                                comp = pd.DataFrame(W.columns.values)
                                perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                                perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                                # Normalise the loadings for each component to be between 0 and 1:
                                H = Normalise(H.transpose())

                                for index, row in H.iterrows():
                                    vimp = row * perm_imp['Permutation_Importance']
                                    if index in d_nmf_theory:
                                        d_nmf_theory[index].append(vimp.sum())
                                    else:
                                        d_nmf_theory[index] = [vimp.sum()]

                    else:
                        for index, row in df.iterrows():
                            print("Evaluating "+ features + " " + model_name+", MAE :" +str(row['MAE']))
                            if features == "All_features":
                                H = Get_H_matrix(X=X, best_alpha=row['nmf__alpha'],
                                                 best_n=row['nmf__n_components'],
                                                 best_solver=row['nmf__solver'],
                                                 random_state=row['dt__random_state'],
                                                 max_iter=max_iter, tol=tol)
                                W = Get_W_matrix(X=X, best_alpha=row['nmf__alpha'],
                                                 best_n=row['nmf__n_components'],
                                                 best_solver=row['nmf__solver'],
                                                 random_state=row['dt__random_state'],
                                                 max_iter=max_iter, tol=tol)
                                nmf_dt= DecisionTreeRegressor(
                                    random_state=row['dt__random_state'],
                                    min_samples_split=row['dt__min_samples_split'],
                                    max_depth=row['dt__max_depth'])

                                nmf_dt.fit(W, y)
                                r = permutation_importance(nmf_dt, W, y, n_repeats=n_repeats, random_state=0,
                                                           scoring="neg_mean_absolute_error")
                                imp = pd.DataFrame(r.importances_mean)
                                comp = pd.DataFrame(W.columns.values)
                                perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                                perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                                # Normalise the loadings for each component to be between 0 and 1:
                                H = Normalise(H.transpose())

                                for index, row in H.iterrows():
                                    vimp = row * perm_imp['Permutation_Importance']
                                    if index in d_nmf_all:
                                        d_nmf_all[index].append(vimp.sum())
                                    else:
                                        d_nmf_all[index] = [vimp.sum()]

                            if features == "Theory_features":
                                X = X[keep_vars]
                                H = Get_H_matrix(X=X, best_alpha=row['nmf__alpha'],
                                                 best_n=row['nmf__n_components'],
                                                 best_solver=row['nmf__solver'],
                                                 random_state=row['dt__random_state'],
                                                 max_iter=max_iter, tol=tol)
                                W = Get_W_matrix(X=X, best_alpha=row['nmf__alpha'],
                                                 best_n=row['nmf__n_components'],
                                                 best_solver=row['nmf__solver'],
                                                 random_state=row['dt__random_state'],
                                                 max_iter=max_iter, tol=tol)
                                nmf_dt = DecisionTreeRegressor(
                                    random_state=row['dt__random_state'],
                                    min_samples_split=row['dt__min_samples_split'],
                                    max_depth=row['dt__max_depth'])

                                nmf_dt.fit(W, y)
                                r = permutation_importance(nmf_dt, W, y, n_repeats=n_repeats, random_state=0,
                                                           scoring="neg_mean_absolute_error")
                                imp = pd.DataFrame(r.importances_mean)
                                comp = pd.DataFrame(W.columns.values)
                                perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                                perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                                # Normalise the loadings for each component to be between 0 and 1:
                                H = Normalise(H.transpose())

                                for index, row in H.iterrows():
                                    vimp = row * perm_imp['Permutation_Importance']
                                    if index in d_nmf_theory:
                                        d_nmf_theory[index].append(vimp.sum())
                                    else:
                                        d_nmf_theory[index] = [vimp.sum()]

                else:
                    for index, row in df.iterrows():
                        print("Evaluating "+ features + " " + model_name+", MAE :" +str(row['MAE']))
                        if features == "All_features":
                            H = Get_H_matrix(X=X, best_alpha=row['nmf__alpha'],
                                             best_n=row['nmf__n_components'],
                                             best_solver=row['nmf__solver'],
                                             random_state=row['rf__random_state'],
                                             max_iter=max_iter, tol=tol)
                            W = Get_W_matrix(X=X, best_alpha=row['nmf__alpha'],
                                             best_n=row['nmf__n_components'],
                                             best_solver=row['nmf__solver'],
                                             random_state=row['rf__random_state'],
                                             max_iter=max_iter, tol=tol)
                            nmf_rf = RandomForestRegressor(
                                random_state=row['rf__random_state'],
                                min_samples_split=row['rf__min_samples_split'],
                                max_depth=row['rf__max_depth'],
                                n_estimators=row['rf__n_estimators'])

                            nmf_rf.fit(W, y.values.ravel())
                            r = permutation_importance(nmf_rf, W, y, n_repeats=n_repeats, random_state=0,
                                                       scoring="neg_mean_absolute_error")
                            imp = pd.DataFrame(r.importances_mean)
                            comp = pd.DataFrame(W.columns.values)
                            perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                            perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                            # Normalise the loadings for each component to be between 0 and 1:
                            H = Normalise(H.transpose())

                            for index, row in H.iterrows():
                                vimp = row * perm_imp['Permutation_Importance']
                                if index in d_nmf_all:
                                    d_nmf_all[index].append(vimp.sum())
                                else:
                                    d_nmf_all[index] = [vimp.sum()]

                        if features == "Theory_features":
                            X = X[keep_vars]
                            H = Get_H_matrix(X=X, best_alpha=row['nmf__alpha'],
                                             best_n=row['nmf__n_components'],
                                             best_solver=row['nmf__solver'],
                                             random_state=row['rf__random_state'],
                                             max_iter=max_iter, tol=tol)
                            W = Get_W_matrix(X=X, best_alpha=row['nmf__alpha'],
                                             best_n=row['nmf__n_components'],
                                             best_solver=row['nmf__solver'],
                                             random_state=row['rf__random_state'],
                                             max_iter=max_iter, tol=tol)
                            nmf_rf = RandomForestRegressor(
                                random_state=row['rf__random_state'],
                                min_samples_split=row['rf__min_samples_split'],
                                max_depth=row['rf__max_depth'],
                                n_estimators=row['rf__n_estimators'])

                            nmf_rf.fit(W, y.values.ravel())
                            r = permutation_importance(nmf_rf, W, y, n_repeats=n_repeats, random_state=0,
                                                       scoring="neg_mean_absolute_error")
                            imp = pd.DataFrame(r.importances_mean)
                            comp = pd.DataFrame(W.columns.values)
                            perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                            perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                            # Normalise the loadings for each component to be between 0 and 1:
                            H = Normalise(H.transpose())

                            for index, row in H.iterrows():
                                vimp = row * perm_imp['Permutation_Importance']
                                if index in d_nmf_theory:
                                    d_nmf_theory[index].append(vimp.sum())
                                else:
                                    d_nmf_theory[index] = [vimp.sum()]

            else:
                if 'DT' in model_name:
                    if '(MF)' in model_name:
                        for index, row in df.iterrows():
                            print("Evaluating "+ features + " " + model_name+", MAE :" +str(row['MAE']))
                            if features == "All_features":
                                dt_mf = DecisionTreeRegressor(
                                    random_state=int(row['random_state']),
                                    min_samples_split=int(row['min_samples_split']),
                                    max_features=row['max_features'],
                                    max_depth=int(row['max_depth']))

                                dt_mf.fit(X, y)
                                r = permutation_importance(dt_mf, X, y, n_repeats=n_repeats, random_state=0,
                                                           scoring="neg_mean_absolute_error")
                                imp = pd.DataFrame(r.importances_mean)
                                comp = pd.DataFrame(X.columns.values)
                                perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                                perm_imp.columns = ['Variable', 'Permutation_Importance']

                                for index, row in perm_imp.iterrows():
                                    v = row['Variable']
                                    if v in d_non_nmf_all:
                                        d_non_nmf_all[v].append(row['Permutation_Importance'])
                                    else:
                                        d_non_nmf_all[v] = row['Permutation_Importance']

                            if features == "Theory_features":
                                X = X[keep_vars]

                                dt_mf = DecisionTreeRegressor(
                                random_state=int(row['random_state']),
                                min_samples_split=int(row['min_samples_split']),
                                max_features=row['max_features'],
                                max_depth=int(row['max_depth']))

                                dt_mf.fit(X, y)
                                r = permutation_importance(dt_mf, X, y, n_repeats=n_repeats, random_state=0,
                                                               scoring="neg_mean_absolute_error")
                                imp = pd.DataFrame(r.importances_mean)
                                comp = pd.DataFrame(X.columns.values)
                                perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                                perm_imp.columns = ['Variable', 'Permutation_Importance']

                                for index, row in perm_imp.iterrows():
                                    v = row['Variable']
                                    if v in d_non_nmf_theory:
                                        d_non_nmf_theory[v].append(row['Permutation_Importance'])
                                    else:
                                        d_non_nmf_theory[v] = row['Permutation_Importance']

                    else:
                        for index, row in df.iterrows():
                            print("Evaluating "+ features + " " + model_name+", MAE :" +str(row['MAE']))
                            if features == "All_features":

                                d_nmf_theory = DecisionTreeRegressor(
                                    random_state=row['random_state'],
                                    min_samples_split=row['min_samples_split'],
                                    max_depth=row['max_depth'])

                                d_nmf_theory.fit(X, y)
                                r = permutation_importance(d_nmf_theory, X, y, n_repeats=n_repeats, random_state=0,
                                                           scoring="neg_mean_absolute_error")
                                imp = pd.DataFrame(r.importances_mean)
                                comp = pd.DataFrame(X.columns.values)
                                perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                                perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                                for index, row in perm_imp.iterrows():
                                    v = row['Variable']
                                    if v in d_non_nmf_all:
                                        d_non_nmf_all[v].append(row['Permutation_Importance'])
                                    else:
                                        d_non_nmf_all[v] = row['Permutation_Importance']

                            if features == "Theory_features":
                                X = X[keep_vars]

                                d_nmf_theory = DecisionTreeRegressor(
                                    random_state=row['random_state'],
                                    min_samples_split=row['min_samples_split'],
                                    max_depth=row['max_depth'])

                                d_nmf_theory.fit(X, y)
                                r = permutation_importance(d_nmf_theory, X, y, n_repeats=n_repeats, random_state=0,
                                                           scoring="neg_mean_absolute_error")
                                imp = pd.DataFrame(r.importances_mean)
                                comp = pd.DataFrame(X.columns.values)
                                perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                                perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                                for index, row in perm_imp.iterrows():
                                    v = row['Variable']
                                    if v in d_non_nmf_theory:
                                        d_non_nmf_theory[v].append(row['Permutation_Importance'])
                                    else:
                                        d_non_nmf_theory[v] = row['Permutation_Importance']

                else:
                    for index, row in df.iterrows():
                        print("Evaluating "+ features + " " + model_name+", MAE :" +str(row['MAE']))
                        if features == "All_features":

                            rf = RandomForestRegressor(
                                random_state=row['random_state'],
                                min_samples_split=row['min_samples_split'],
                                max_depth=row['ax_depth'],
                                n_estimators=row['n_estimators'])

                            rf.fit(X, y.values.ravel())
                            r = permutation_importance(rf, X, y, n_repeats=n_repeats, random_state=0,
                                                       scoring="neg_mean_absolute_error")
                            imp = pd.DataFrame(r.importances_mean)
                            comp = pd.DataFrame(X.columns.values)
                            perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                            perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                            for index, row in perm_imp.iterrows():
                                v = row['Variable']
                                if v in d_non_nmf_all:
                                    d_non_nmf_all[v].append(row['Permutation_Importance'])
                                else:
                                    d_non_nmf_all[v] = row['Permutation_Importance']

                        if features == "Theory_features":
                            X = X[keep_vars]

                            rf = RandomForestRegressor(
                                random_state=row['random_state'],
                                min_samples_split=row['min_samples_split'],
                                max_depth=row['ax_depth'],
                                n_estimators=row['n_estimators'])

                            rf.fit(X, y.values.ravel())
                            r = permutation_importance(rf, X, y, n_repeats=n_repeats, random_state=0,
                                                       scoring="neg_mean_absolute_error")
                            imp = pd.DataFrame(r.importances_mean)
                            comp = pd.DataFrame(X.columns.values)
                            perm_imp = pd.concat([comp, imp], axis=1, ignore_index=True)
                            perm_imp.columns = ['Basis_num', 'Permutation_Importance']

                            for index, row in perm_imp.iterrows():
                                v = row['Variable']
                                if v in d_non_nmf_theory:
                                    d_non_nmf_theory[v].append(row['Permutation_Importance'])
                                else:
                                    d_non_nmf_theory[v] = row['Permutation_Importance']


save_path = "Outputs/Rashomon_Set/Variable_Importance/"
results_nmf_model_T = pd.DataFrame.from_dict(d_nmf_theory)
results_nmf_model_T=results_nmf_model_T.sum()
results_nmf_model_T.sort_values(ascending=False, inplace=True)
results_nmf_model_T = pd.DataFrame(results_nmf_model_T)
results_nmf_model_T.to_csv(save_path+'Rash_Variable_Importance_NMF_Model_Theory.csv')

results_nmf_model_A = pd.DataFrame.from_dict(d_nmf_all)
results_nmf_model_A=results_nmf_model_A.sum()
results_nmf_model_A.sort_values(ascending=False, inplace=True)
results_nmf_model_A = pd.DataFrame(results_nmf_model_A)
results_nmf_model_A.to_csv(save_path+'Rash_Variable_Importance_NMF_Model_All.csv')

results_non_nmf_model_T = pd.DataFrame.from_dict(d_non_nmf_theory, orient='index')
if results_non_nmf_model_T.shape[1] > 1:
    results_non_nmf_model_T=results_non_nmf_model_T.sum()
    results_non_nmf_model_T.sort_values(ascending=False, inplace=True)
results_non_nmf_model_T.sort_values(by=0, ascending=False, inplace=True)
results_non_nmf_model_T.to_csv(save_path+'Rash_Variable_Importance_NON_NMF_Model_Theory.csv')

results_non_nmf_model_A = pd.DataFrame.from_dict(d_non_nmf_all, orient='index')
if results_non_nmf_model_A.shape[1] > 1:
    results_non_nmf_model_A=results_non_nmf_model_A.sum()
    results_non_nmf_model_A.sort_values(ascending=False, inplace=True)
if results_non_nmf_model_A.shape[1] > 0:
    results_non_nmf_model_A.sort_values(by=0, ascending=False, inplace=True)
    results_non_nmf_model_A.to_csv(save_path+'Rash_Variable_Importance_NON_NMF_Model_All.csv')

# combine 2 dfs Theory:
all_results_T = results_non_nmf_model_T.merge(results_nmf_model_T, left_index=True, right_index=True)
all_results_T.columns=["Non_NMF_Importance", "NMF_Importance"]
all_results_T['Combined_Importance']= all_results_T["Non_NMF_Importance"]+ all_results_T["NMF_Importance"]
all_results_T.sort_values(by='Combined_Importance', inplace=True, ascending=False)
all_results_T.to_csv(save_path+'Rash_Variable_Importance_allmodels_THEORY.csv')

print("Sum NMF Importance Theory = " + str(all_results_T["NMF_Importance"].sum()))
print("Sum Non NMF Importance Thoery = " + str(all_results_T["Non_NMF_Importance"].sum()))

# combine 2 dfs All features:
if (results_non_nmf_model_A.shape[1] > 0) and (results_nmf_model_A.shape[1] > 0):
    all_results_A = results_non_nmf_model_A.merge(results_nmf_model_A, left_index=True, right_index=True)
    all_results_A.columns=["Non_NMF_Importance", "NMF_Importance"]
    all_results_A['Combined_Importance']= all_results_A["Non_NMF_Importance"]+ all_results_A["NMF_Importance"]
    all_results_A.sort_values(by='Combined_Importance', inplace=True, ascending=False)
    all_results_A.to_csv(save_path+'Rash_Variable_Importance_allmodels_ALLFEATURES.csv')

    print("Sum NMF Importance Theory = " + str(all_results_A["NMF_Importance"].sum()))
    print("Sum Non NMF Importance Thoery = " + str(all_results_A["Non_NMF_Importance"].sum()))
else:
    all_results_A = results_non_nmf_model_A.merge(results_nmf_model_A, left_index=True, right_index=True)
    all_results_A.to_csv(save_path + 'Rash_Variable_Importance_allmodels_ALLFEATURES.csv')




# todo: have one for theory and one for all features

print("done!")