import pandas as pd
from Model_Classes.NON_NMF_MODEL import Lasso, max_features_is_n_features,\
    max_features_as_param, RF, PLS
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
# --------------------------------------------------------------------------

# Run models ALL features:
results_path = "Results/All_features/"

# set fixed params:
n_folds = 70
print_every_nrows = 100
rash_threshold = 0.27
test = False
seeds = range(1, 51)

# # PLS regression
#
# results_name = "PLS_"
# Pls = PLS(X=X, y=y, seed=None, max_iter=1000,
#             results_name=results_path + results_name + "Results.txt")
#
# Pls.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=print_every_nrows,
#          test=test)

# # Lasso
# results_name = "LASSO_"
# lasso = Lasso(X=X, y=y, seed=seeds,
#             results_name=results_path + results_name + "Results.txt")
#
# lasso.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=print_every_nrows,
#          test=test)

# DT
# results_name = "DT_"
# dt = max_features_is_n_features(X=X, y=y, seed=seeds,
#         results_name=results_path + results_name + "Results.txt")
#
# dt.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=print_every_nrows,
#          test=test)
#
# # DT_MF
# results_name = "DT_MF_"
# dt_mf = max_features_as_param(X=X, y=y, seed=seeds,
#         results_name=results_path + results_name + "Results.txt")
#
# dt_mf.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=print_every_nrows,
#          test=test)
#
# # RF
# results_name = "RF_"
# rf = RF(X=X, y=y, seed=seeds,
#         results_name=results_path + results_name + "Results.txt")
#
# rf.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=print_every_nrows,
#          test=test)

# ----------------------------------------------------------------------------------

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

# PLS regression
results_name = "PLS_"
Pls = PLS(X=X, y=y, seed=None, max_iter=1000,
          results_name=results_path + results_name + "Results.txt")

Pls.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
         rash_threshold=rash_threshold,
         rash_results_name=results_path + results_name + "Rash_results.csv",
         n_folds=n_folds,
         print_every_nrows=print_every_nrows,
         test=test)
# # Lasso
# results_name = "LASSO_"
# lasso = Lasso(X=X, y=y, seed=seeds,
#             results_name=results_path + results_name + "Results.txt")
#
# lasso.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=print_every_nrows,
#          test=test)

# DT
results_name = "DT_"
dt = max_features_is_n_features(X=X, y=y, seed=seeds,
        results_name=results_path + results_name + "Results.txt")

dt.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
         rash_threshold=rash_threshold,
         rash_results_name=results_path + results_name + "Rash_results.csv",
         n_folds=n_folds,
         print_every_nrows=print_every_nrows,
         test=test)

# DT_MF
results_name = "DT_MF_"
dt_mf = max_features_as_param(X=X, y=y, seed=seeds,
        results_name=results_path + results_name + "Results.txt")

dt_mf.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
         rash_threshold=rash_threshold,
         rash_results_name=results_path + results_name + "Rash_results.csv",
         n_folds=n_folds,
         print_every_nrows=print_every_nrows,
         test=test)

# # RF
# results_name = "RF_"
# rf = RF(X=X, y=y, seed=seeds,
#         results_name=results_path + results_name + "Results.txt")
#
# rf.LOOCV(verbose_name=results_path + results_name + "Verbose.txt",
#          rash_threshold=rash_threshold,
#          rash_results_name=results_path + results_name + "Rash_results.csv",
#          n_folds=n_folds,
#          print_every_nrows=print_every_nrows,
#          test=test)
