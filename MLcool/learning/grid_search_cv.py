#!/usr/bin/env python3

import numpy as np
from pandas import read_csv
from time import time
from joblib import Parallel, delayed, cpu_count

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .kernels import SimpleKernel


'''
params:
    - features      :   [
                            [[ ... ],[ ... ], ..., [ ... ]],
                            [[ ... ],[ ... ], ..., [ ... ]],
                                          ...              ,
                            [[ ... ],[ ... ], ..., [ ... ]]
                        ]
    . target        : [[1],[-8], ... [90]]
    - train_splits  : [[ ... ], [ ... ], ..., [ ... ]]
    - test_splits   : [[ ... ], [ ... ], ..., [ ... ]]
    - valid_split   : [ ... ]
    - hyper_ranges  : [(-2, 2, 10), (-15, 2, 100), (-0.1, 0.1, 5)]
    - hyper_model   : (-5, 0, 800)

    - model         : "krr"
    - kernel_type   : "laplacian"
    - n_jobs        : 6
    - loss          : " MAE"
    - relaunch      : False
    - output        : "results_test1"
    - verbose       : False, True, 0, 1, 2


shape of arrays:
    - features      : (n_samples, n_interactions, n_features,)
    - target        : (n_samples, 1,)
    - train_splits  : (n_splits, n_train_samples,)
    - test_splits   : (n_splits, n_test_samples,)
    - valid_split   : (n_valid_samples,)
    - hyper_ranges  : (n_interactions, 3,)
    - hyper_model   : (3,)

constrains: 
    - n_test_samples + n_train_samples + n_valid_samples = n_samples
    - In hyper_range and hyper_model (lower_bound, upper_bound, step) IN LOG SCALE
'''
    

class CVSampler(object):
    def __init__(self, params):

        # Convert all entries of dict() "params" to CVSampler attributes
        for k in params.keys():
            value = params[k]
            try:
                attr = literal_eval(value)

            except:
                attr = value

            setattr(self, k, attr)

        self.csv_file       = '{}.csv'.format(self.output)

        self.log10a_space   = np.linspace(*self.hyper_model)
        self.a_space        = np.power(10., self.log10a_space)
        self.n_splits       = len(self.train_splits)
        self.n_g            = len(self.hyper_ranges)
        self.n_a            = len(self.a_space)
        self.time_track     = None

        self._arange_log10g()
        self._check_init()


    def __call__(self):
        self._sample()

    
    def _arange_log10g(self):
        if self.n_g == 1:
            self.all_log10g = np.linspace(*this_g).reshape(-1,1)

        else:
            ranges = np.array([np.linspace(*this_g) for this_g in self.hyper_ranges], dtype = object)
            self.all_log10g = self._arange(ranges)


        self.n_iter = len(self.all_log10g)


    def _arange(self, all_g, current_g = None, l = None):
        if current_g is None:
            current_g   = [[g] for g in all_g[0]]

            return self._arange(all_g, current_g, 1)

        elif l != self.n_g:
            new_g = []
            for gp in current_g:
                for gn in all_g[l]:
                    new_g.append([*gp, gn])

            current_g = new_g

            return self._arange(all_g, current_g, l+1)

        return np.array(current_g)


    def _update_best(self, log10g, log10a, ts_loss, it):
        if self.best is None:
            self.best = [log10g, log10a, ts_loss, it]

        else:
            if self.best[2] > ts_loss:
                self.best = [log10g, log10a, ts_loss, it]


    # MAIN LOOP FUNCTION:
    ##############################################################################
    def _sample(self):
        for idx in range(self.first_it, self.n_iter):
            log10g = self.all_log10g[idx]

            # DO SOME PRINTS:
            if self.verbose >= 1:
                self.print_cv_headder(idx, log10g)

            # Loss of model evaluated at log10g and all model's hyperparameters (a)
            best_tr_loss, best_ts_loss, best_vl_loss, best_log10a = self.evaluate_grid_point(log10g)

            # UPDATE BEST RESULT
            self._update_best(log10g, best_log10a, best_ts_loss[-1], idx)

            # MORE PRINTS: CV SUMMARY
            if self.verbose >= 1:
                self.print_cv_summary(log10g, best_log10a, best_tr_loss, best_ts_loss, best_vl_loss)

            # PRINT IN CSV FILE
            self._flush(idx, log10g, best_log10a, best_tr_loss, best_ts_loss, best_vl_loss)

    
    def evaluate_grid_point(self, log10g):
        # Undo the log10
        g                       = np.power(10., log10g)

        # Track time of the evaluation
        t_cv_ini                = time()
        ctr_tr, ctr_ts, ctr_vl  = ([] for _ in range(3))
        a_time                  = []

        # Losses along splits:
        point_tr_loss, point_ts_loss, point_vl_loss = ([] for _ in range(3))

        # The validation split is always fixed thus this can be done outside the loop:
        vl_features = self.features[self.valid_split,:,:]
        vl_target   = self.target[self.valid_split,:]

        for ind, (tr_index, ts_index) in enumerate(zip(self.train_splits, self.test_splits)):

            # Training and test features (and targets) for ind-th split:
            tr_features = self.features[tr_index,:,:]
            ts_features = self.features[ts_index,:,:]

            tr_target   = self.target[tr_index,:]
            ts_target   = self.target[ts_index,:]

            # Create the kernel for train, test and validation splits
            kernel      = self.kernel(g, self.kernel_type)
            t0          = time()
            tr_k        = kernel.get(tr_features, n_jobs = self.n_jobs)
            t1          = time()
            ctr_tr.append(t1-t0)

            t0          = time()
            ts_k        = kernel.get(tr_features, ts_features, n_jobs = self.n_jobs)
            t1          = time()
            ctr_ts.append(t1-t0)

            t0          = time()
            vl_k        = kernel.get(tr_features, vl_features, n_jobs = self.n_jobs)
            t1          = time()
            ctr_vl.append(t1-t0)

            # Clean memory
            del tr_features, ts_features

            #import matplotlib.pyplot as plt
            #plt.imshow(tr_k)
            #plt.show()
            #plt.imshow(ts_k)
            #plt.show()
            #plt.imshow(vl_k)
            #plt.show()


            # Iterate along the model's inner hyper-parameter
            # arange differently depending on the number of jobs specified
            t0 = time()
            if self.n_jobs == 1:
                # Serial version
                tr_loss, ts_loss, vl_loss = self.fit_and_predict(
                        self.a_space,
                        tr_k, tr_target, 
                        ts_k, ts_target, 
                        vl_k, vl_target
                        )

            else:
                # Parallel version
                parallel_jobs = []
                for split_of_a in np.array_split(self.a_space, self.n_jobs):
                    d = delayed(self.fit_and_predict)
                    d = d(
                            split_of_a,
                            tr_k, tr_target,
                            ts_k, ts_target,
                            vl_k, vl_target
                            )

                    parallel_jobs.append(d)

                r                           = Parallel(n_jobs = self.n_jobs)(parallel_jobs)
                tr_loss, ts_loss, vl_loss   = np.concatenate(r, axis = 1)


            t1 = time()
            a_time.append(t1-t0)

            # Best tracked loss (last index, -1) in test for this split
            best_loss_index = np.argmin(ts_loss[:,-1])
            best_log10a     = self.log10a_space[best_loss_index]

            best_tr_loss    = tr_loss[best_loss_index]
            best_ts_loss    = ts_loss[best_loss_index]
            best_vl_loss    = vl_loss[best_loss_index]

            # DO SOME PRINTS:
            if self.verbose >= 2:
                self.print_losses(best_tr_loss, best_ts_loss, best_vl_loss, best_log10a, ind)

            # Accumulate for all splits:
            point_tr_loss.append(tr_loss)
            point_ts_loss.append(ts_loss)
            point_vl_loss.append(vl_loss)


        # DO AVERAGES OF ALL SPLITS:
        point_tr_loss = np.mean(point_tr_loss, axis = 0)
        point_ts_loss = np.mean(point_ts_loss, axis = 0)
        point_vl_loss = np.mean(point_vl_loss, axis = 0)

        best_loss_index = np.argmin(point_ts_loss[:,-1])
        best_log10a     = self.log10a_space[best_loss_index]

        best_tr_loss    = point_tr_loss[best_loss_index]
        best_ts_loss    = point_ts_loss[best_loss_index]
        best_vl_loss    = point_vl_loss[best_loss_index]

        # Time consumption:
        t_cv_fin                        = time()
        t_cv                            = t_cv_fin - t_cv_ini
        ctr_tr, ctr_ts, ctr_vl, a_time  = (np.mean(t) for t in [ctr_tr, ctr_ts, ctr_vl, a_time])
        self.time_tracker               = [t_cv, ctr_tr, ctr_ts, ctr_vl, a_time]

        return best_tr_loss, best_ts_loss, best_vl_loss, best_log10a


    def fit_and_predict(self, a_space, tr_k, tr_target, ts_k, ts_target, vl_k, vl_target):

        tr_loss, ts_loss, vl_loss = ([] for _ in range(3))

        for a in a_space:
            model = self.model(kernel = 'precomputed', alpha = a)
            model.fit(tr_k, tr_target)

            # For training set
            prediction      = model.predict(tr_k)
            losses          = self.evaluate_losses(tr_target, prediction)
            tr_loss.append(losses)

            # For test set
            prediction      = model.predict(ts_k)
            losses          = self.evaluate_losses(ts_target, prediction)
            ts_loss.append(losses)

            # For validation set
            prediction      = model.predict(vl_k)
            losses          = self.evaluate_losses(vl_target, prediction)
            vl_loss.append(losses)


        # Just convert to numpy array ...
        tr_loss, ts_loss, vl_loss = (np.array(split) for split in [tr_loss, ts_loss, vl_loss])
        

        # Returns a tuple of 3 lists of shape (n_a_space, 4)
        return tr_loss, ts_loss, vl_loss


    def evaluate_losses(self, target, prediction):
        mae_loss        = calc_MAE(target, prediction)
        rmse_loss       = calc_RMSE(target, prediction)
        r2_loss         = calc_R2(target, prediction)
        tracked_loss    = self.loss(target, prediction)

        return [mae_loss, rmse_loss, r2_loss, tracked_loss]


    def _flush(self, it, log10g, best_log10a, best_tr_loss, best_ts_loss, best_vl_loss):
        _, _, _, best_it = self.best

        
        values      = '{:.4e},'*self.n_g
        values      = values[:len(values)-1]

        string      = '{},{},{:.4e},{},'.format(it, best_it,  best_log10a, values.format(*log10g))
        string     += '{:.4e},{:.4e},{:.4e},'.format(*best_tr_loss)
        string     += '{:.4e},{:.4e},{:.4e},'.format(*best_ts_loss)
        string     += '{:.4e},{:.4e},{:.4e}\n'.format(*best_vl_loss)

        with open(self.csv_file, 'a') as fl:
            fl.write(string)


    def print_cv_headder(self, idx, v):
        string1 = 'BEGIN CROSS-VALIDATION (f_calls = {}/{})'.format(idx, self.n_iter)
        string2 = '**********************************************************************'
        string3 = 'Function Params: log({})=({})\n'

        n       = list(range(1,self.n_g+1))
        names   = 'g{},'*self.n_g
        names   = names[:len(names)-1]
        
        values  = '{:.5f},'*self.n_g
        values  = values[:len(values)-1]

        string3 = string3.format(names, values)
        string3 = string3.format(*n,*v)

        print(string1)
        print(string2)
        print(string3)


    def print_losses(self, tr, ts, vl, log10a, ind):
        print(' CV split (nº {}/{})'.format(ind+1, self.n_splits))
        print(' (Best log(alpha) for this split is {:.4f})'.format(log10a))
        print(' ....................................................')
            
        if self.verbose >= 3:
            self._print_losses(tr, ts, vl)


    def _print_losses(self, tr, ts, vl):
        string = '{} {:4}: {:.5f}'
        for ind, l in enumerate(['MAE', 'RMSE', 'R2']):
            print(string.format('  Train     ', l, tr[ind]))
            print(string.format('  Test      ', l, ts[ind]))
            print(string.format('  Validation', l, vl[ind]))
            if ind != 2:
                print('  ************************')

            else:
                print('', flush = True)


    def print_cv_summary(self, log10g, best_log10a, tr, ts, vl):
        lines = '----------------------------------------------------------------------'

        t_cv, ctr_tr, ctr_ts, ctr_vl, a_time    = self.time_tracker

        n       = list(range(1,self.n_g+1))
        names   = 'g{},'*self.n_g
        names   = names[:len(names)-1]
        
        values  = '{:.5f},'*self.n_g
        values  = values[:len(values)-1]

        string0 = '\n CV SUMMARY:\n'

        string1 = '  Parameters: log({})=({})\n'
        string1 = string1.format(names, values)
        string1 = string1.format(*n,*log10g)

        string2 = '  Best Cross Validated log(alpha) found: {:.5f}\n'
        string2 = string2.format(best_log10a)

        string3 = '  Average time (for CV-split) to calculate the Kernel with {} cores (seconds):'
        string3 = string3.format(self.n_jobs)

        string4 = '    Train:\t{:.2f}\n    Test:\t{:.2f}\n    Validation:\t{:.2f}\n'
        string4 = string4.format(ctr_tr, ctr_ts, ctr_vl)

        string5 = '  Average time (for CV-split) to fit and predict {:3} values of alpha: {:.2f}\n'
        string5 = string5.format(self.n_a, a_time)


        log10g, log10a, ts_loss, it = self.best
        values                      = '{:.5f},'*self.n_g
        values                      = values[:len(values)-1]

        string6 = '  CURRENT BEST RESULT (f_call = {}):'
        string6 = string6.format(it)

        string7 = '  log(alpha), log({}), LOSS = {},({}),{}\n'
        string7 = string7.format(names, '{:.5f}', values, '{:.5f}')
        string7 = string7.format(*n, log10a, *log10g, ts_loss)

        string8 = '  TOTAL CV TIME CONSUMED: {:.2f} seconds\n'
        string8 = string8.format(t_cv)

        print(lines)
        print(lines)
        print(string0)
        print(string1)
        print(string2)
        self._print_losses(tr, ts, vl)
        print(string3)
        print(string4)
        print(string5)
        print(string6)
        print(string7)
        print(string8)
        print(lines)
        print(lines)
        print('\n', flush=True)


    def _check_init(self):
        if self.model == 'krr':
            model_name  = self.model
            self.model  = KernelRidge

        else:
            print('Model attribute value error. ')
            print('The model can only be "krr" for now')
            print('Sampling with krr.')
            print()

            model_name  = self.model
            self.model  = KernelRidge


        if self.kernel_type in ['rbf', 'laplacian']:
            self.kernel = SimpleKernel

        else:
            print('kernel_type must be one of these options:')
            print('"laplacian" or "rbf"')
            print('Please, choose a correct one')
            print('Exit.')
            exit()


        self.n_jobs = cpu_count() if self.n_jobs == -1 else self.n_jobs
        metrics     = {
                'MAE'   : calc_MAE,
                'RMSE'  : calc_RMSE,
                'R2'    : calc_R2
                }

        if self.loss in metrics.keys():
            loss_name = self.loss
            self.loss = metrics[self.loss]

        else:
            print('Loss function must be one of these options:')
            print(metrics.keys())
            print('Please, choose a correct one')
            print('Exit.')
            exit()


        if not self.relaunch:
            self.best       = None
            self.first_it   = 0


            n           = list(range(1,self.n_g+1))
            names       = 'log_g{},'*self.n_g
            names       = names[:len(names)-1]

            headder     = 'it,current_best_it,log_a,{},'
            headder    += 'tr_mae,tr_rmse,tr_r2,'
            headder    += 'ts_mae,ts_rmse,ts_r2,'
            headder    += 'vl_mae,vl_rmse,vl_r2\n'
            headder     = headder.format(names)
            headder     = headder.format(*n)

            sep1 = '#\n'
            sep2 = '###############################################################\n'


            with open(self.csv_file, 'w') as fl:
                fl.write(sep1)
                fl.write(sep2)
                fl.write(sep2)
                fl.write('# nº of splits                : {}\n'.format(self.n_splits))
                fl.write('# nº of validation samples    : {}\n'.format(len(self.valid_split)))
                fl.write('# g range                     : {}\n'.format(self.hyper_ranges))
                fl.write('# a range                     : {}\n'.format(self.hyper_model))
                fl.write('# model                       : {}\n'.format(model_name))
                fl.write('# kernel                      : {}\n'.format(self.kernel_type))
                fl.write('# loss function               : {}\n'.format(loss_name))
                fl.write(sep2)
                fl.write(sep2)
                fl.write(sep1)

                fl.write(headder)


        else:
            df              = read_csv(self.csv_file, comment = '#')
            last_row        = df.iloc[[-1]]

            self.first_it   = last_row['it'].values[0] + 1
            best_iter       = last_row['current_best_it'].values[0]

            best_iter_df    = df.iloc[[best_iter]]
            best_log10a     = best_iter_df['log_a'].values[0]
            best_log10g     = best_iter_df.filter(like='log_g').values[0]

            target_loss     = 'ts_{}'.format(loss_name.lower())
            best_loss       = best_iter_df[target_loss].values[0]

            self.best       = [best_log10g, best_log10a, best_loss, best_iter]



        if self.features.shape[1] != len(self.hyper_ranges):
            print('Size of features vector does not match')
            print('the number of hyperparameters.')
            print('Exit')
            exit()


        if isinstance(self.verbose, bool):
            self.verbose = 3 if self.verbose else 0

        elif not isinstance(self.verbose, int):
            print('Invalid verbosity level.')
            print('verbosity set to "3"')
            self.verbose = 3



def calc_MAE(target, prediction):
    return mean_absolute_error(target, prediction)


def calc_RMSE(target, prediction):
    return np.sqrt(mean_squared_error(target, prediction))


def calc_R2(target, prediction):
    return r2_score(target, prediction)



