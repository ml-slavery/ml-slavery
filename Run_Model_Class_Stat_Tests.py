import pandas as pd
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from Functions.Preprocessing.Normalise import Normalise
from sklearn.cross_decomposition import PLSRegression
from scipy import stats

# Improvements over best-of-class models for linear,
# lasso and partial least squares regression
# were also all significant (Bonferroni corrected) across the board.

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

cv = LeaveOneOut()
m1_score = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
mae_m1 = abs(m1_score).mean()
print("SKLearn MAE score: "+str(mae_m1))
# --------------------------------------------------------------------------

param_dict = {"NMF_LM":
                  {'nmf__alpha': 1,
                   'nmf__n_components': 5,
                   'nmf__random_state': 14,
                   'nmf__solver': 'cd'},

              "LASSO": {'alpha': 0.01,
                        'random_state': 15,
                        'tol': 0.1},
              "PLS":
                  {'n_components': 8,
                   'tol': 1}
}

#  NMF->LM with all features:

nmf_lm = Pipeline([
        ("nmf", NMF(init="nndsvd", max_iter=350, tol=0.005,
                    alpha=param_dict["NMF_LM"].get('nmf__alpha'),
                    n_components=param_dict["NMF_LM"].get('nmf__n_components'),
                    random_state=param_dict["NMF_LM"].get('nmf__random_state'),
                    solver= param_dict["NMF_LM"].get('nmf__solver'))),
        ("lm", LinearRegression()
        )])

best_nmf_lm_score = cross_val_score(nmf_lm, X, y, cv=cv, scoring="neg_mean_absolute_error")

# non-parametric (not normal) paired (same data) t-test = Wilcoxon Signed-Rank test:

print("Best NMF->LM MAE score: " + str(abs(best_nmf_lm_score).mean()))
stat, p = stats.wilcoxon(x=m1_score, y=best_nmf_lm_score, zero_method='wilcox', alternative='two-sided')
print("m1 and best NMF->LM (all features), p = {:f}".format(p))

# Lasso with theory features:

lasso = Pipeline([
        ("lasso", linear_model.Lasso(normalize=False, max_iter=10000,
                                    selection='random',
                                    alpha=param_dict["LASSO"].get('alpha'),
                                    tol=param_dict["LASSO"].get('tol'),
                                    random_state=param_dict["LASSO"].get('random_state')))])

best_lasso_score = cross_val_score(lasso, X, y, cv=cv, scoring="neg_mean_absolute_error")

print("Best LASSO MAE score: " + str(abs(best_lasso_score).mean()))
stat, p = stats.wilcoxon(x=m1_score, y=best_lasso_score, zero_method='wilcox', alternative='two-sided')
print("m1 and best LASSO (theory features), p = {:f}".format(p))

# PLS with all features:

pls = Pipeline([
    ("PLS", PLSRegression(n_components=param_dict["PLS"].get('n_components'),
                          scale=False,
                          max_iter=350,
                          tol=param_dict["PLS"].get('tol'),
                          copy=False))])

best_pls_score = cross_val_score(pls, X, y, cv=cv, scoring="neg_mean_absolute_error")

print("Best PLS MAE score: " + str(abs(best_pls_score).mean()))
stat, p = stats.wilcoxon(x=m1_score, y=best_pls_score, zero_method='wilcox', alternative='two-sided')
print("m1 and best PLS (all features), p = {:f}".format(p))

print('done!')


