
import pandas as pd
from Model_Classes.NMF_MODEL import NMF_LM, NMF_RF, max_features_is_n_features, max_features_as_param
from Functions.Preprocessing.Normalise import Normalise


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

# Run models ALL features:
results_path = "Results/All_features/"

# set fixed params:
n_folds = 70
print_every_nrows = 100
rash_threshold = 0.27
test = False
seeds = range(1, 51)
nmf_max_iter = 350
tol = 0.005

# NMF->LM
results_name = "NMF_LM_"
lm = NMF_LM(X=X, y=y, seed=seeds, nmf_max_iter=nmf_max_iter, tol=tol,
            results_name=results_path + results_name + "Results.txt")

lm.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
         rash_threshold=rash_threshold,
         rash_results_name=results_path + results_name + "Rash_results.csv",
         n_folds=n_folds,
         print_every_nrows=print_every_nrows,
         test=test)

# NMF->DT
results_name = "NMF_DT_"
tree_mf = max_features_is_n_features(X=X,y=y, seed=seeds, nmf_max_iter=nmf_max_iter,
                                     tol=tol, results_name=results_path+results_name+"Results.txt")

tree_mf.LOOCV(verbose_name=results_path+results_name+"Verbose.txt",
                   rash_threshold=rash_threshold,
                   rash_results_name=results_path + results_name + "Rash_results.csv",
                   n_folds=n_folds,
                   print_every_nrows=print_every_nrows,
                   test=test)
#
# NMF->DT(MF)
# results_name = "NMF_DT_MF_"
# tree_mf = max_features_as_param(X=X,y=y, seed=seeds, nmf_max_iter=nmf_max_iter, tol=tol,
#                                 results_name=results_path+results_name+"Results.txt")
#
# tree_mf.LOOCV(verbose_name=results_path+results_name+"Verbose.txt",
#                    rash_threshold=rash_threshold,
#                    rash_results_name=results_path + results_name + "Rash_results.csv",
#                    n_folds=n_folds,
#                    print_every_nrows=print_every_nrows,
#                    test=test)

# # NMF->RF
# results_name = "NMF_RF_"
# rf = NMF_RF(X=X, y=y, seed=seeds, nmf_max_iter=nmf_max_iter, tol=tol,
#             results_name=results_path + results_name + "Results.txt")
#
# rf.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=500,
#          test=test)
# #
# Run models THEORY features:
results_path = "Results/Theory_features/"

keep_vars = ['KOF_Globalis','Work_rightCIRI','Trade_open','GDPpc',
 'Armedcon','Poverty', 'Stunting_u5s','Undernourish','Wasting_u5s','Maternal_mort',
 'Neonatal_mort','Literacy_15_24yrs','F_school','ATMs','Child_lab','Unemploy','Infrastruct',
 'Internet_use','Broadband','Climate_chg_vuln','CPI','Minority_rule','Freemv_M',
 'Freemv_F','Free_discuss','Soc_powerdist','Democ',
 'Sexwrk_condom','Sexwrk_Syphilis','AIDS_Orph','Rape_report',
 'Rape_enclave','Phys_secF','Gender_equal']
X = X[keep_vars]

# # NMF->LM
# results_name = "NMF_LM_"
# lm = NMF_LM(X=X, y=y, seed=seeds, nmf_max_iter=nmf_max_iter, tol=tol,
#             results_name=results_path + results_name + "Results.txt")
#
# lm.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=print_every_nrows,
#          test=test)
#
# # # NMF->DT
# results_name = "NMF_DT_"
# tree_mf = max_features_is_n_features(X=X,y=y, seed=seeds, nmf_max_iter=nmf_max_iter,
#                                      tol=tol, results_name=results_path+results_name+"Results.txt")
#
# tree_mf.LOOCV(verbose_name=results_path+results_name+"Verbose.txt",
#                    rash_threshold=rash_threshold,
#                    rash_results_name=results_path + results_name + "Rash_results.csv",
#                    n_folds=n_folds,
#                    print_every_nrows=print_every_nrows,
#                    test=test)
#
# NMF->DT(MF)
results_name = "NMF_DT_MF_"
tree_mf = max_features_as_param(X=X,y=y, seed=seeds, nmf_max_iter=nmf_max_iter, tol=tol,
                                results_name=results_path+results_name+"Results.txt")

tree_mf.LOOCV(verbose_name=results_path+results_name+"Verbose.txt",
                   rash_threshold=rash_threshold,
                   rash_results_name=results_path + results_name + "Rash_results.csv",
                   n_folds=n_folds,
                   print_every_nrows=print_every_nrows,
                   test=test)
#
# # NMF->RF
# results_name = "NMF_RF_"
# rf = NMF_RF(X=X, y=y, seed=seeds, nmf_max_iter=nmf_max_iter, tol=tol,
#             results_name=results_path + results_name + "Results.txt")
#
# rf.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=500,
#          test=test)
print('done')

