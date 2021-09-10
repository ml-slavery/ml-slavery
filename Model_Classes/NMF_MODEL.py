import numpy as np
import pandas as pd
import datetime
import sys
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

class NMF_Model(object):

    def __init__(self, X, y, seed,
                 cv=70,
                 n_comp=[5, 6, 7], # 2-8 components were first tried before narrowing down the parameter search space to between 5 and 7 after results from an initial run
                 alpha=[0.5, 1, 2],
                 solver=["cd"]
                 ):
        self.results_name = None
        self.X = X
        self.y = y
        self.seed = seed
        self.cv = cv
        self.n_comp = n_comp
        self.alpha = alpha
        self.solver = solver
        self.pipe = None

        self.param_grid = {"nmf__solver": self.solver,
                           "nmf__n_components": self.n_comp,
                           "nmf__alpha": self.alpha
                           }

        self.TESTparam_grid = {"nmf__solver": self.solver,
                               "nmf__n_components": [7],
                               "nmf__alpha": [2]}

    def change_y(self):
        if isinstance(self.y, pd.DataFrame):
            self.y = np.array(self.y['SLAVERY'])
        return

    def print_start(self):
        num_fits = self.num_fits
        num_iterations = self.num_iterations
        grid_print = self.grid_print
        f = self.f
        print("The number of fits =" + str(num_fits), file=f)
        print("The number of iterations =" + str(num_iterations), file=f)
        print("Params iterating over  = " + str(list(grid_print.items())), file=f)
        f.flush()

    def print_best_params(self, error, params):
        print("UPDATE MAE: " + str(error), file=self.f)
        print("UPDATE Best params = " + str(params), file=self.f)
        self.f.flush()

    def print_nrows(self, p, params, param_start_time_n):
        print("Fit num: " + str(p) + "/" + str(self.num_fits) +
              "..." + str(np.round((p / self.num_fits) * 100, 0)) + "%", file=self.f)
        param_set_time_n = datetime.now() - param_start_time_n
        print("Time taken: " + str(param_set_time_n), file=self.f)
        print("Current params = " + str(params), file=self.f)
        self.f.flush()

    def print_end_results(self, params, runtime):
        print("==== RUN: " + str(datetime.now()) + " ===="
              , file=open(self.results_name, "w"))
        print("RESULTS MAE: " + str(params['Error']), file=open(self.results_name, "a"))
        print("RESULTS Best params = " + str(params['Params']), file=open(self.results_name, "a"))
        print("RUNTIME: " + str(runtime), file=open(self.results_name, "a"))
        sys.stdout.close()

    def update_rash_params(self, error, params):
        self.rash_params[error] = params

    def update_best_params(self, error, params):
        self.Results['Params'] = params
        self.Results['Error'] = error

    def LOOCV(self, verbose_name, rash_threshold, rash_results_name, n_folds, print_every_nrows,
              test):
        n = self.X.shape[0]
        self.rash_params = {}
        self.Results = {}
        current_mae = 1
        p = 0

        # If string index's, need to be re-set:
        self.X.reset_index(inplace=True, drop=True)
        self.y.reset_index(inplace=True, drop=True)

        grid_start = datetime.now()
        self.f = open(verbose_name, "w")
        print(str(grid_start), file=self.f)
        param_start_time_n = datetime.now()

        if test == True:
            self.grid_print = self.TESTparam_grid.items()
            self.grid = ParameterGrid(self.TESTparam_grid)
        else:
            self.grid_print = self.param_grid
            self.grid = ParameterGrid(self.param_grid)

        self.num_fits = len(list(self.grid))
        self.num_iterations = self.num_fits * n_folds
        self.print_start()

        for param in list(self.grid):
            err_fold_list = []
            self.set_pipeline(**param)

            for i in range(0, n_folds):
                X_test_fold = pd.DataFrame(self.X.iloc[i, :]).transpose()
                X_train_folds = self.X.drop(i, axis=0)
                y_test_fold = self.y.iloc[i, :]
                y_train_folds = self.y.drop(i, axis=0)

                m1 = self.pipe.fit(X_train_folds, y_train_folds.values.ravel())
                pred = m1.predict(X_test_fold)
                err = abs(y_test_fold.values - pred)
                err_fold_list.append(err[0])
                self.f.flush()

            param_mae = pd.Series(err_fold_list).mean()
            self.f.flush()

            if param_mae <= rash_threshold:
                self.update_rash_params(error=param_mae, params=param)

            if param_mae <= current_mae:
                current_mae = param_mae
                self.update_best_params(error=param_mae, params=param)
                self.print_best_params(error=param_mae, params=param)

            # Print where at in grid search every n_rows:
            p = p + 1
            if p % print_every_nrows == 0:
                self.print_nrows(p=p, params=param, param_start_time_n=param_start_time_n)

                # Save Rashomon Set parameters every nrows:
                Rashomon_models = pd.DataFrame.from_dict(self.rash_params, orient='index')
                with open(rash_results_name, 'w') as r_file:
                        Rashomon_models.to_csv(r_file)
                        r_file.flush()

            # Save final set of Rashomon Set params:
            if p == self.num_fits:
                Rashomon_models = pd.DataFrame.from_dict(self.rash_params, orient='index')
                with open(rash_results_name, 'w') as r_file:
                        Rashomon_models.to_csv(r_file)
                        r_file.flush()

        # Calculate and print runtime:
        Runtime = datetime.now() - grid_start
        print("Total Run time: " + str(Runtime), file=self.f)
        self.f.flush()
        self.f.close()

        self.print_end_results(params=self.Results, runtime=Runtime)

        return

    def set_pipeline(self, **kwargs):
        return


class NMF_LM(NMF_Model):

    def __init__(self, X, y, seed, nmf_max_iter, tol, results_name):
        super().__init__(X, y, seed)
        self.results_name = results_name
        self.tol = tol
        self.nmf_max_iter = nmf_max_iter
        self.seed = seed

        self.param_grid["nmf__random_state"] = self.seed
        return

    def set_pipeline(self, nmf__solver, nmf__n_components, nmf__alpha, nmf__random_state):
        self.pipe = Pipeline([
            ("nmf", NMF(init="nndsvd", solver=nmf__solver, max_iter=self.nmf_max_iter,
                        tol=self.tol,
                        n_components=nmf__n_components, alpha=nmf__alpha,
                        verbose=0, random_state=nmf__random_state)),
            ("lm", LinearRegression())])


class NMF_DT(NMF_Model):

    def __init__(self, X, y, seed, nmf_max_iter, tol, results_name,
                 min_samples_split=[2, 3, 4],
                 max_depth=[3, 4, 5, 6]):
        super().__init__(X, y, seed)
        self.results_name = results_name
        self.nmf_max_iter = nmf_max_iter
        self.tol = tol

        self.param_grid["dt__random_state"] = seed
        self.param_grid["dt__min_samples_split"] = min_samples_split
        self.param_grid["dt__max_depth"] = max_depth

        self.TESTparam_grid["dt__random_state"] = self.seed
        self.TESTparam_grid["dt__min_samples_split"] = [3]
        self.TESTparam_grid["dt__max_depth"] = [4]

        return


class max_features_is_n_features(NMF_DT):
    def __init__(self, X, y, seed, nmf_max_iter, tol, results_name):
        super().__init__(X, y, seed, nmf_max_iter, tol, results_name)
        return

    def set_pipeline(self, nmf__solver, nmf__n_components, nmf__alpha, dt__random_state,
                     dt__min_samples_split, dt__max_depth):
        self.pipe = Pipeline([
            ("nmf", NMF(init="nndsvd", solver=nmf__solver, max_iter=self.nmf_max_iter,
                        tol=self.tol,
                        n_components=nmf__n_components, alpha=nmf__alpha,
                        verbose=0, random_state=dt__random_state)),
            ("dt", DecisionTreeRegressor(random_state=dt__random_state,
                                         min_samples_split=dt__min_samples_split,
                                         max_depth=dt__max_depth))])


class max_features_as_param(NMF_DT):
    def __init__(self, X, y, seed, nmf_max_iter, tol, results_name, max_features=[0.3, 0.5, 0.7]):
        super().__init__(X, y, seed, nmf_max_iter, tol, results_name)
        self.results_name = results_name
        self.nmf_max_iter = nmf_max_iter
        self.tol = tol

        self.param_grid['dt__max_features'] = max_features
        self.TESTparam_grid['dt__max_features'] = [0.5]

        return

    def set_pipeline(self, nmf__solver, nmf__n_components, nmf__alpha, dt__random_state,
                     dt__min_samples_split, dt__max_depth, dt__max_features):
        self.pipe = Pipeline([
            ("nmf", NMF(init="nndsvd", solver=nmf__solver, max_iter=self.nmf_max_iter,
                        tol=self.tol,
                        n_components=nmf__n_components, alpha=nmf__alpha,
                        verbose=0, random_state=dt__random_state)),
            ("dt", DecisionTreeRegressor(random_state=dt__random_state,
                                         min_samples_split=dt__min_samples_split,
                                         max_depth=dt__max_depth,
                                         max_features=dt__max_features))])
        return


class NMF_RF(NMF_Model):

    def __init__(self, X, y, seed, nmf_max_iter, tol, results_name,
                 min_samples_split=[2, 3],
                 max_depth=[4, 5],
                 max_features=[0.3, 0.5],
                 n_estimators=[30, 50]
                 ):
        super().__init__(X, y, seed)

        self.results_name = results_name
        self.nmf_max_iter = nmf_max_iter
        self.tol = tol

        self.param_grid["rf__random_state"] = self.seed
        self.param_grid["rf__min_samples_split"] = min_samples_split
        self.param_grid["rf__max_depth"] = max_depth
        self.param_grid["rf__max_features"] = max_features
        self.param_grid["rf__n_estimators"] = n_estimators

        self.TESTparam_grid["rf__random_state"] = self.seed
        self.TESTparam_grid["rf__min_samples_split"] = [3]
        self.TESTparam_grid["rf__max_depth"] = [4]
        self.TESTparam_grid["rf__max_features"] = [0.3]
        self.TESTparam_grid["rf__n_estimators"] = [60]

        return

    def set_pipeline(self, nmf__solver, nmf__n_components, nmf__alpha, rf__random_state,
                     rf__min_samples_split, rf__max_depth, rf__max_features, rf__n_estimators):
        self.pipe = Pipeline([
            ("nmf", NMF(init="nndsvd", solver=nmf__solver, max_iter=self.nmf_max_iter,
                        tol=self.tol,
                        n_components=nmf__n_components, alpha=nmf__alpha,
                        verbose=0, random_state=rf__random_state)),
            ("rf", RandomForestRegressor(random_state=rf__random_state,
                                         min_samples_split=rf__min_samples_split,
                                         max_depth=rf__max_depth,
                                         max_features=rf__max_features,
                                         n_estimators=rf__n_estimators))])
        return



