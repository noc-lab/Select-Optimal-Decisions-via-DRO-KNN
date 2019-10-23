"""
This script provide a distributed robust regression model. Borrow
largely from the Huber regression script in sklearn
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/huber.py

Henghui Zhu
"""
import sys
sys.path.append('/home/rchen15/prescription/run_pres')

import numpy as np
from scipy import optimize, sparse

from sklearn.linear_model.base import BaseEstimator, RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length
from sklearn.utils import axis0_safe_slice
from sklearn.utils.extmath import safe_sparse_dot


def solve_drr(X, y, reg_l1, reg_l2):
    import cvxpy as cvx
    import cvxopt

    num_sample, num_dim = X.shape

    y = np.reshape(y, -1)
    beta = cvx.Variable(num_dim)

    obj = reg_l2 * cvx.norm([-beta, 1], 2) + reg_l1 * cvx.norm(beta, 1) + cvx.norm(X*beta - y, 1) / num_sample
    p = cvx.Problem(cvx.Minimize(obj))
    p.solve()
    if p.status == cvx.OPTIMAL or p.status == cvx.OPTIMAL_INACCURATE:
        return np.array(beta.value)


def _drr_loss_and_gradient(w, X, y, reg_l2, reg_l1):
    """Returns the distributionally robust regression loss and the gradient.
    Parameters
    ----------
    w : ndarray, shape (n_features + 1,)
        Feature vector.
        w[:n_features] gives the coefficients
        w[-1] gives the scale factor and if the intercept is fit w[-2]
        gives the intercept factor.
    X : ndarray, shape (n_samples, n_features)
        Input data.
    y : ndarray, shape (n_samples,)
        Target vector.
    reg_l2 : float
        l2 regularization parameter of the estimator.
    reg_l1 : float
        l1 regularization parameter of the estimator.
    sample_weight : ndarray, shape (n_samples,), optional
        Weight assigned to each sample. (not applicable right now)
    Returns
    -------
    loss : float
        regression loss.
    gradient : ndarray, shape (len(w))
        Returns the derivative of the regression loss with respect to each
        coefficient, intercept and the scale as a vector.
    """
    # TODO: 1. add sample_weight; 2. adapt to sparse matrix;
    # 3. avoid use reshape by analyzing dimension of each variables
    beta_norm = np.sqrt(np.sum(np.square(w))+1)
    #beta_norm = np.sum(np.square(w))
    l2_grad_reg = reg_l2 * 2 * w
    #l2_grad_reg = reg_l2 * w / (np.sqrt(np.sum(np.square(w))+1))
    l1_grad_reg = reg_l1 * np.sign(w)

    sign_err = safe_sparse_dot(X, w) - y

    grad_err_pre = np.multiply(np.sign(np.expand_dims(sign_err, 1)), X)
    grad_err = np.mean(grad_err_pre, axis=0).reshape(w.shape)

    loss = np.mean(np.abs(sign_err)) + reg_l2 * beta_norm + \
           reg_l1 * np.sum(np.abs(w))

    #return loss, l2_grad_reg + l1_grad_reg + grad_err
    return loss


# TODO: add document
class DistributionallyRobustRegressor(LinearModel, RegressorMixin, BaseEstimator):
    def __init__(self, reg_l2=0.1, reg_l1=0.1, max_iter=5000,
                 warm_start=False, fit_intercept=True, tol=1e-05,
                 solver='scipy'):
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.solver = solver

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples, 1)
            Target vector relative to X.
        sample_weight : array-like, shape (n_samples,)
            Weight given to each sample.
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(
            X, y, copy=False, accept_sparse=['csr'], y_numeric=True)

        # ensure y's shape is valid
        # y = np.reshape(y, [-1, 1])

        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            check_consistent_length(y, sample_weight)
        else:
            sample_weight = np.ones_like(y)

        if self.fit_intercept:
            arg_X = np.concatenate([X, np.ones([X.shape[0], 1])], axis=1)
        else:
            arg_X = X

        if self.solver == 'scipy':
            if self.warm_start and hasattr(self, 'coef_'):
                parameters = np.concatenate(
                    (self.coef_, [self.intercept_]))
            else:
                if self.fit_intercept:
                    parameters = np.random.rand(X.shape[1] + 1)
                else:
                    parameters = np.random.rand(X.shape[1])

            parameters = np.reshape(parameters, [-1, 1])

            # Type Error caused in old versions of SciPy because of no
            # maxiter argument ( <= 0.9).
#            try:
#                parameters, f, dict_ = optimize.fmin_l_bfgs_b(
#                    _drr_loss_and_gradient, parameters,
#                    args=(arg_X, y, self.reg_l2, self.reg_l1, sample_weight),
#                    approx_grad = 1, maxiter=self.max_iter, pgtol=self.tol,
#                    iprint=0)
#            except TypeError:
#                parameters, f, dict_ = optimize.fmin_l_bfgs_b(
#                    _drr_loss_and_gradient, parameters,
#                    args=(arg_X, y, self.reg_l2, self.reg_l1, sample_weight),approx_grad = 1)
#
#            if dict_['warnflag'] == 2:
#                raise ValueError("DRR convergence failed:"
#                                 " l-BFGS-b solver terminated with %s"
#                                 % dict_['task'].decode('ascii'))
#            self.n_iter_ = dict_.get('nit', None)
            res = optimize.minimize(_drr_loss_and_gradient, parameters, args=(arg_X, y, self.reg_l2, self.reg_l1), method = 'Nelder-Mead', options={'disp': False, 'maxiter': self.max_iter}) 
            parameters = res.x

        elif self.solver == 'cvx':
            parameters = solve_drr(arg_X, y, self.reg_l1, self.reg_l2)

            if parameters is None:
                raise Exception("Solver did not converge!")
        else:
            raise NotImplementedError


        parameters = parameters.reshape([-1])

        if self.fit_intercept:
            self.intercept_ = parameters[-1]
        else:
            self.intercept_ = 0.0
        self.coef_ = parameters[:X.shape[1]]

        # Outlier analysis
        # residual = np.abs(
        #     y - safe_sparse_dot(X, self.coef_) - self.intercept_)
        # self.outliers_ = residual > self.scale_ * self.epsilon

        return self
