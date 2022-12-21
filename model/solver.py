"""
Solver
======
Several solvers needed to solve Matrix GLM problems and its transfer learning problems.

Available Solvers
-----------------
nuclear_solver
    Solving matrix GLM coefficient with nuclear-norm penalty.\n
    This function can be used to solve `Gauss` model and `Binary Logistic` model.

lasso_solver
    Solving matrix GLM coefficient with l1-norm penalty.\n
    This function can be used to solve `Gauss` model and `Binary Logistic` model.

multinomial_nuclear_solver
    Solving matrix GLM coefficient with nuclear-norm penalty.\n
    This function can be used to solve `Multinomial Logistic` model.

multinomial_lasso_solver
    Solving matrix GLM coefficient with l1-norm penalty.\n
    This function can be used to solve `Multinomial Logistic` model.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso, Ridge

from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from typing import Optional, Union, Callable, Any

# Extrapolation for gradient descent point
def extrapolation(model, alpha1 : float, alpha2 : float):
    """
    extrapolation(model : Union[MatrixClassifier, MatrixRegressor],
                  alpha1 : float, alpha2 : float) -> np.ndarray
        Do extrapolation to accelerate gradient descent algorithm convergence
    """
    return model.coef_ + (alpha1 - 1) / alpha2 * (model.coef_ - model.coef_pre)

# Singular values optimization
def SingularValsOptim(a : np.ndarray, _lambda : float, _delta : float):
    b = (a - _lambda * _delta)
    # soft thresholding the singular values
    b[b <= 1e-15] = 0
    return b

# Coordinate Descent algorithm for binary lasso problem
def CDN_binary_lasso(X : np.ndarray, y : np.ndarray, 
                     _lambda : float, max_iter : int, eps : Optional[float]=1e-4,
                     offset : Optional[np.ndarray]=None, 
                     sigma : Optional[float]=0.9, beta : Optional[float]=0.9):
    """
    Coordinate Descent algorithm for binary classification problem with l1-norm penalty.

    Parameters
    ----------
    X : np.ndarray
        The observed covariates.
    y : np.ndarray
        The observed response.
    _lambda : float
        The penalty coefficient.
    max_iter : int
        The maximum number of iterations.
    eps : float, default = `1e-4`
        The threshold for judging algorithm convergence.
    offset : np.ndarray, default = `None`
        In the transfer learning problem, provide `offset` items `X @ transfer_coef` for fine-tuning.
    sigma : float, default = `0.9`
        The upper bound constant of line search.
    beta : float, default = `0.9`
        The discount factor of line search step.\n
        The learning rate will set to `{1,beta,beta^2,...}` for line search.

    Return
    ----------
    coef : np.ndarray
        The estimated model coefficients.
    historyObj : list
        The updating process of objective function.
    """
    # The calculation of probability in likelihood and loss will use exp_Logits 
    # exp_logits = exp{-(offset + X @ coef)}
    # to reduce the calculation cost of inner product X @ coef

    # compute negative log-likelihood
    def LogLikelihood(exp_logits : np.ndarray):
        proba = 1 / (1 + exp_logits)
        loss = -1 * np.mean(y * np.log(proba + 1e-16) + (1 - y) * np.log(1 - proba + 1e-16))
        return loss
    # calculate objective value
    def objective(exp_logits : np.ndarray, coef : np.ndarray):
        loss = LogLikelihood(exp_logits)
        penalty = _lambda * np.abs(coef.flatten()).sum()
        return loss + penalty
    # calculate the first-order partial derivative
    def first_order_derivative(
        proba : np.ndarray, X : np.ndarray, y : np.ndarray, j : int):
        return -np.mean((y - proba)*X[:,j])
    # calculate the second-order partial derivative
    def second_order_derivative(
        proba : np.ndarray, X : np.ndarray, j : int):
        return np.mean(proba*(1-proba)*X[:,j]**2)
    
    # init coef
    n, p = X.shape[0], X.shape[1]
    coef = np.zeros(p)
    # compute offset
    offset = offset if offset is not None else np.zeros(n)

    # calculate exponential logits
    exp_logits = np.exp(-(offset + X@coef))

    # store loss
    historyObj = []
    historyObj.append(objective(exp_logits,coef))

    # Initialize the feature set participating in the optimization
    feats_set = set(list(range(p)))

    # measure the violation of the optimality condition
    v = np.zeros(p)

    # init global step and err
    step, err = 0, 1
    while step < max_iter and err > eps:

        # randomly arrange feature order
        feats = list(feats_set)
        np.random.shuffle(feats)

        # Iteration on each feature
        for j in feats:
            # calculate derivative
            proba = 1 / (1 + exp_logits) # sigmoid transform
            first_d = first_order_derivative(proba,X,y,j)
            second_d = second_order_derivative(proba,X,j)

            # moving variables out of the optimization process hierarchically
            if step > 0:
                if coef[j] == 0 and (-1 + M < first_d) and (first_d < 1 - M):
                    feats_set.remove(j)
                    continue

            # the optimal Newton direction
            if first_d + _lambda <= second_d*coef[j]:
                d = -(first_d + _lambda) / second_d
            elif first_d - _lambda >= second_d*coef[j]:
                d = -(first_d - _lambda) / second_d
            else:
                d = -coef[j]
            
            # update measure the violation of the optimality condition
            if coef[j] > 0:
                v[j] = np.abs(first_d + _lambda)
            elif coef[j] < 0:
                v[j] = np.abs(first_d - _lambda)
            else:
                v[j] = max(first_d - _lambda,-first_d - _lambda,0)

            # line search
            k = 0
            L_w = LogLikelihood(exp_logits=exp_logits) # log-likelihood before update
            while k < 25: # cyclic protection
                t = beta**k
                z = t*d
                exp_logits_z = exp_logits * np.exp(-(z*X[:,j])) # exp_logits after update
                L_z = LogLikelihood(exp_logits=exp_logits_z) # log-likelihood after update

                # diff = g_j(z) - g_j(0) = g_j(z) = \lambda * (|w_j + z| - |w_j|) + L(w +z*e_j) - L(w)
                # ensure that the decrease of g_j(z) meets the requirements
                # this calculation of diff can be improved by approximation method 
                # to improve the calculation speed
                diff = _lambda*(abs(coef[j] + z) - abs(coef[j])) + L_z - L_w
                # diff =  _lambda*(abs(coef[j] + z) - abs(coef[j])) + first_d*z + 0.5*second_d*z**2
                if diff <= sigma*t*(first_d*d + _lambda*np.abs(coef[j]+d)-_lambda*np.abs(coef[j])):
                    break
                k += 1
            
            # update coef and exponential logits
            coef[j] += z
            exp_logits = exp_logits_z
        
        # update step
        step += 1
        # determine how aggressive we remove variables
        M = np.max(v) / n
        # update ojective value
        historyObj.append(objective(exp_logits,coef))
        err = np.abs(historyObj[-1] - historyObj[-2])
            
    return coef, historyObj

# Coordinate Descent algorithm for multinomial lasso problem
def CDN_multinomial_lasso(X : np.ndarray, y : np.ndarray, 
                          _lambda : float, max_iter : int, eps : Optional[float]=1e-4,
                          offset : Optional[np.ndarray]=None, 
                          sigma : Optional[float]=0.9, beta : Optional[float]=0.9):
    """
    Coordinate Descent algorithm for multinomial classification problem with l1-norm penalty.

    Parameters
    ----------
    X : np.ndarray
        The observed covariates.
    y : np.ndarray
        The observed response.
    _lambda : float
        The penalty coefficient.
    max_iter : int
        The maximum number of iterations.
    eps : float, default = `1e-4`
        The threshold for judging algorithm convergence.
    offset : np.ndarray, default = `None`
        In the transfer learning problem, provide `offset` items `X @ transfer_coef.T` for fine-tuning.\n
        offset has shape `(n,K)`, `K` is the number of classes.
    sigma : float, default = `0.9`
        The upper bound constant of line search.
    beta : float, default = `0.9`
        The discount factor of line search step.\n
        The learning rate will set to `{1,beta,beta^2,...}` for line search.

    Return
    ----------
    coef : np.ndarray
        The estimated model coefficients.
    historyObj : list
        The updating process of objective function.
    """
    # transform y into labels
    if len(y.shape) == 1:
        y = OneHotEncoder().fit_transform(y.reshape(-1,1)).toarray()



def nuclear_solver(model, X : np.ndarray, y : np.ndarray,
                          delta : float,
                          max_steps : int, max_steps_epoch : int,
                          epsilon : float):
    """
    Solving matrix GLM coefficient with nuclear-norm penalty.
    This function can be used to solve `Gauss` model and `Binary Logistic` model.
    """
    # init variables
    historyObj = []
    alpha = [0, 1]
    step = 0
    # init objective value
    historyObj.append(model.objective(X,y,model.coef_))
    # init singular value
    _, singular_vals, _ = np.linalg.svd(model.coef_,full_matrices=False)

    model.coef_pre = model.coef_ # save coef

    while True:
        # update global step
        step += 1
        # compute extrapolation
        S = extrapolation(model,alpha[-2],alpha[-1])
        
        local_step = 0 # init local step
        while True:
            # update local step
            local_step += 1

            # compute Atmp
            Atmp = S - delta * model.gradients(S,X,y)
            # do SVD
            U, a, V = np.linalg.svd(Atmp,full_matrices=False)
            # solve sub-optimazation problems
            b = SingularValsOptim(a,model._lambda,delta)
            # compute Btmp
            Btmp = U @ np.diag(b) @ V
            # update delta
            delta /= 2
            # Armijo line search
            BtmpObj = model.objective(X,y,Btmp)
            if local_step >= max_steps_epoch \
                or BtmpObj <= model.approximate(Btmp,S,X,y,delta,Atmp) \
                or delta < 1e-16:
                break
        
        # update coef
        model.coef_pre = model.coef_
        # Force Descent Condition
        if BtmpObj <= historyObj[-1]:
            # update coef and singular values
            model.coef_ = Btmp
            singular_vals = b
            stepObj = BtmpObj
        else:
            stepObj = historyObj[-1]
        # add history
        historyObj.append(stepObj)

        # Update alpha
        alpha.append((1 + np.sqrt(1 + (2*alpha[-1])**2)) / 2)

        # Objective Value Converge
        if step >= max_steps or abs(historyObj[-1] - historyObj[-2]) < epsilon:
            break

    return historyObj, singular_vals, step

def lasso_solver(model, X : np.ndarray, y : np.ndarray, max_steps : int, transfer : bool):
    """
    Solving matrix GLM coefficient with l1-norm penalty.\n
    This function can be used to solve `Gauss` model and `Binary Logistic` model.
    """
    # flatten
    p, q = X.shape[1], X.shape[2]
    X = np.reshape(X, (X.shape[0],-1))

    if transfer and model.task == "classification":
        # compute offset
        transfer_coef = model.transfer_coef.copy()
        transfer_coef = transfer_coef.flatten()
        offset = X @ transfer_coef
        # solve coef using coordinate descent algorithm
        coef,_ = CDN_binary_lasso(X=X,y=y,_lambda=model._lambda,offset=offset,max_iter=max_steps)
    else:
        # build model
        if model.task == "regression":
            lasso_model = Lasso(alpha=model._lambda,max_iter=max_steps,fit_intercept=False,warm_start=False)
        elif model.task == "classification":
            lasso_model = LogisticRegression(penalty="l1",C=1/model._lambda,solver="saga",max_iter=max_steps,fit_intercept=False,warm_start=False)
        # train
        lasso_model.fit(X=X,y=y)
        # fetch coef
        coef = lasso_model.coef_.copy()

    coef = np.reshape(coef, (p,q))
    model.coef_ = coef
    # compute singular values
    _, singular_vals, _ = np.linalg.svd(model.coef_,full_matrices=False)
    model.singular_vals = singular_vals

    return

def multinomial_nuclear_solver(model, X : np.ndarray, y : np.ndarray,
                               delta : float,
                               max_steps : int, max_steps_epoch : int,
                               epsilon : float):
    """
    Solving matrix GLM coefficient with nuclear-norm penalty.\n
    This function can be used to solve `Multinomial Logistic` model.
    """
    # init variables
    historyObj = []
    alpha = [0, 1]
    step = 0
    p, q = X.shape[1], X.shape[2]

    # save singular values
    singular_vals = np.zeros((model.n_class,min(p,q)))

    # encode label
    if len(y.shape) == 1:
        encoder = OneHotEncoder(dtype=np.int64)
        label = encoder.fit_transform(y.reshape(-1,1)).toarray()
    else:
        label = y

    # init objective value
    historyObj.append(model.objective(X,label,model.coef_))
    # init singular value
    for c in model.classes:
        _, singular_val, _ = np.linalg.svd(model.coef_[c],full_matrices=False)
        singular_vals[c] = singular_val

    model.coef_pre = model.coef_ # save coef

    while True:
        # update global step
        step += 1
        # compute extrapolation
        S = extrapolation(model,alpha[-2],alpha[-1])
        
        local_step = 0 # init local step
        while True:
            # update local step
            local_step += 1

            # compute Atmp
            Atmp = S - delta * model.gradients(S,X,label)
            Btmp = np.zeros_like(Atmp)
            # store singular values
            optim_singularVals = np.zeros_like(singular_vals)
            # do SVD
            for c in model.classes:
                U, a, V = np.linalg.svd(Atmp[c],full_matrices=False)
                # solve sub-optimazation problems
                optim_singularVals[c] = SingularValsOptim(a,model._lambda,delta)
                # compute Btmp
                Btmp[c] = U @ np.diag(optim_singularVals[c]) @ V
            # update delta
            delta /= 2
            # Armijo line search
            BtmpObj = model.objective(X,label,Btmp)
            if local_step >= max_steps_epoch \
                or BtmpObj <= model.approximate(Btmp,S,X,label,delta,Atmp) \
                or delta < 1e-16:
                break
        
        # update coef
        model.coef_pre = model.coef_
        # Force Descent Condition
        if BtmpObj <= historyObj[-1]:
            # update coef and singular values
            model.coef_ = Btmp
            singular_vals = optim_singularVals
            stepObj = BtmpObj
        else:
            stepObj = historyObj[-1]
        # add history
        historyObj.append(stepObj)

        # Update alpha
        alpha.append((1 + np.sqrt(1 + (2*alpha[-1])**2)) / 2)

        # Objective Value Converge
        if step >= max_steps or abs(historyObj[-1] - historyObj[-2]) < epsilon:
            break
    
    return historyObj, singular_vals, step


def multinomial_lasso_solver(model, X : np.ndarray, y : np.ndarray, max_steps : int, transfer : bool):
    """
    Solving matrix GLM coefficient with l1-norm penalty.\n
    This function can be used to solve `Multinomial Logistic` model.
    """
    try:
        assert(model.task == "classification")
    except:
        raise ValueError("only support multiple classification problems!")
    # flatten
    p, q = X.shape[1], X.shape[2]
    X = np.reshape(X, (X.shape[0],-1))
    
    # when transfer we need set warm_start=True and debias coef
    if transfer:
        lasso_model = LogisticRegression(
                penalty="l1",C=1/model._lambda,solver="saga",multi_class="multinomial",
                max_iter=max_steps,fit_intercept=False,warm_start=True)
        transfer_coef = model.transfer_coef.copy()
        transfer_coef = np.reshape(transfer_coef, (model.n_class,-1))
        lasso_model.coef_ = transfer_coef
    else:
        lasso_model = LogisticRegression(
                penalty="l1",C=1/model._lambda,solver="saga",multi_class="multinomial",
                max_iter=max_steps,fit_intercept=False,warm_start=False)

    # train
    lasso_model.fit(X=X,y=y)
    # fetch coef and assign to model
    if transfer:
        coef = lasso_model.coef_.copy() - transfer_coef
    else:
        coef = lasso_model.coef_.copy()
    coef = np.reshape(coef, (model.n_class,p,q))
    model.coef_ = coef

    # compute singular values
    singular_vals = np.zeros((model.n_class,min(p,q)))
    for c in model.classes:
        _, singular_val, _ = np.linalg.svd(model.coef_[c],full_matrices=False)
        singular_vals[c] = singular_val
    model.singular_vals = singular_vals

    return

def multinomial_lasso_solver(model, X : np.ndarray, y : np.ndarray, max_steps : int, transfer : bool):
    """
    Solving matrix GLM coefficient with l1-norm penalty.\n
    This function can be used to solve `Multinomial Logistic` model.
    """
    try:
        assert(model.task == "classification")
    except:
        raise ValueError("only support multinomial logistic classification problems!")
    # flatten
    p, q = X.shape[1], X.shape[2]
    X = np.reshape(X, (X.shape[0],-1))
    
    # when transfer we need set warm_start=True and debias coef
    if transfer:
        # compute offset
        transfer_coef = model.transfer_coef.copy()
        transfer_coef = np.reshape(transfer_coef,(model.n_class,-1))
        offset = X @ transfer_coef.T
        coef, _ = CDN_multinomial_lasso(X=X,y=y,_lambda=model._lambda,offset=offset,max_iter=max_steps)
    else:
        lasso_model = LogisticRegression(
                penalty="l1",C=1/model._lambda,solver="saga",multi_class="multinomial",
                max_iter=max_steps,fit_intercept=False,warm_start=False)
        # train
        lasso_model.fit(X=X,y=y)
        # fetch coef
        coef = lasso_model.coef_.copy()
    
    # fetch coef and assign to model
    coef = np.reshape(coef, (model.n_class,p,q))
    model.coef_ = coef

    # compute singular values
    singular_vals = np.zeros((model.n_class,min(p,q)))
    for c in model.classes:
        _, singular_val, _ = np.linalg.svd(model.coef_[c],full_matrices=False)
        singular_vals[c] = singular_val
    model.singular_vals = singular_vals

    return