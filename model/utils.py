from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso, Ridge

from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from typing import Optional, Union, Callable, Any
from joblib import delayed, Parallel
from copy import deepcopy

# compute BIC
def BIC(y_true : np.ndarray, y_pred : np.ndarray, var : float, _lambda : float,
        singular_vals : np.ndarray, singular_lse : np.ndarray, tau : float, 
        p1 : int, p2 : int) -> float:
    """
    BIC(y_true : np.ndarray, y_pred : np.ndarray, var : float,
        singular_vals : np.ndarray, singular_lse : np.ndarray, tau : float) -> float
        Compute Bayesian Information Criterion `BIC` for fitted model.

    Parameters
    ----------
    y_true : np.ndarray
        Observations of label `y`
    y_pred : np.ndarray
        Predicted label `y_pred`
    var : float
        The estimated variance of `epsilon` in regression model
    _lambda : float
        The penalty parameter of the matrix regression model
    singular_vals : np.ndarray
        The singular values of fitted model coef
    singular_lse : np.ndarray
        The singular values of coef obtained by least squares estimate(`Ridge()`)
    tau : float
        The penalty parameter of `Ridge()` method
    p1 : int
        The number of rows for coef
    p2 : int
        The number of columns for coef
    
    Return
    ----------
    bic : float
        The Bayesian Information Criterion `BIC` under penalty parameter `_lambda`
    """
    # sum of square error
    SSE = np.sum((y_true - y_pred)**2)

    # compute effective degree of freedom
    df = 0
    q = np.sum(singular_vals > 0) # the rank of the fitted coef
    p = min(p1,p2)
    for i in range(p):
        ratio = singular_lse[i] * ((1 + tau)*singular_lse[i] - _lambda) / (1 + tau)
        # init
        tmp = 0
        for j in range(p1):
            if j != i:
                tmp += 1 / (singular_lse[i]**2 - singular_lse[j]**2)
        for j in range(p2):
            if j != i:
                tmp += 1 / (singular_lse[i]**2 - singular_lse[j]**2)
        # update df
        if (1 + tau)*singular_lse[i] - _lambda > 0:
            df += 1 + ratio * tmp
        else:
            break

    # df = q*(p1 + p2) - q**2
    bic = SSE / var  + np.log(len(y_true)) * df
    return bic

# estimate lipschitz constant L
def estimate_Lipschitz(X : np.ndarray, task : Optional[str]='regression') -> float:
    """
    estimate_Lipschitz(X : np.ndarray) -> float
        estimate Lipschitz constant L for `d^2 loss(B)` in order to init delta
    
    Parameters
    ----------
    X : np.ndarray
        Observations with shape `(B,P,Q)`
    task : str, optional
        Task of the model, one of `'regression'` and `'classification'`\n
        default is `'regression'`
    
    Return
    ----------
    L : float
        Estimated Lipschitz constant L
    """
    # for Regression problem with scaled Y and identical link function
    L = np.linalg.norm(X,ord='fro',axis=(1,2))
    L = np.mean(L**2)
    
    return L if task == 'regression' else L / 4

def batch_mat_prod(X : np.ndarray, coef : np.ndarray) -> np.ndarray:
    """
    batch_mat_prod(self, X : np.ndarray, coef : np.ndarray) -> np.ndarray
        compute batch matrix inner product
    """
    # batch size
    B = X.shape[0]
    # dimension
    P, Q = X.shape[1], X.shape[2]
    return np.reshape(X,(B,P*Q)) @ coef.flatten()

# do sigmoid transform
def sigmoid(x : np.ndarray) -> np.ndarray:
    """
    sigmoid(x : np.ndarray) -> np.ndarray
        do `Sigmoid` transformation on logits
    """
    return 1 / (1 + np.exp(-x))

# do softmax transform
def softmax(logits : np.ndarray) -> np.ndarray:
    """
    softmax(logits : np.ndarray) -> np.ndarray
        do `Softmax` transformation on logits
    """
    exp_logits = np.exp(logits)
    proba = exp_logits / np.sum(exp_logits,axis=1,keepdims=True)
    return proba

# compute logits
def Logits(X : np.ndarray, coef : np.ndarray) -> np.ndarray:
    """
    Logits(X : np.ndarray, coef : np.ndarray) -> np.ndarra
        compute logits from given covariates `X` and `coef`
    """
    # number of classes
    K = coef.shape[0]
    # init logits
    logits = np.zeros((X.shape[0],K))
    for k in range(K):
        logits[:,k] = batch_mat_prod(X,coef[k])
    
    return logits

# estimate tau of Ridge() estimator when using BIC
def estimate_tau():
    ...

def plot_confusion_matrix(y_true, y_pred, labels, figsize):
    """
    plot_confusion_matrix(y_true, y_pred)
        绘制混淆矩阵
        
    Parameters
    ----------
    y_true : np.ndarray
        数据的真实标签
    y_pred : np.ndarray
        模型的预测结果
    labels : list
        各个类别的含义
    """
    import itertools

    acc = accuracy_score(y_true, y_pred)
    mat = confusion_matrix(y_true, y_pred)
    print("accuracy: %.4f"%(acc))
    
    # 绘制混淆矩阵
    fig = plt.figure(figsize=figsize,dpi=100)
    plt.imshow(mat,cmap=plt.cm.Blues)
    
    thresh = mat.max() / 2
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        # 在每个位置添加上样本量
        plt.text(j, i, mat[i, j],
                 horizontalalignment="center",
                 color="white" if mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.xticks(range(mat.shape[0]),labels)
    plt.yticks(range(mat.shape[0]),labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def multi_task(func : Callable, param_list : list, n_job : int=-1, verbose : int=1) -> list:
    """
    multi_task(func : Callable, param_list : list, n_job : int=-1, verbose : int=1) -> list
        The multi process auxiliary function.\n 
        Pass in a function `func` and its required parameter list `param_List`,\n 
        using multi process acceleration to get calculation results.\n
    
    Parameters
    ----------
    func : Callable
        Callable object that performs the calculation.
    param_list : list of dict
        List of parameters required for `func` calculation.\n
        If multiple parameters need to be passed in, List of dict is recommended, \n
        each dict represents the parameter passed to func when a process is running.\n 
        Dict consists of the signature of the parameter and the value to be used to form a `key: value` pair.
    n_job : int, default = `1`
        Number of multiple processes.\n 
        For details, see the parameter description of joblib.Parallel
    verbose : int, default = `1`
        Tracking level of multi process progress.\n
        For details, see the parameter description of joblib.Parallel.
    
    Return
    ----------
    result : list
        Under each group of parameters, the calculation results of each process \n
        will be saved in a list and returned.
    
    How to use
    ----------
    Suppose the Callable object `func` to be calculated has the following calling forms: \n

      >>> result = func(param1, param2, param3, ...)
    
    In order to track the output for different parameters with the results of multi process auxiliary functions, 
    the following auxiliary function `func_helper` can be constructed: \n
      
      >>> def func_helper(param1, param2, param3, ...):
      >>>     res = {
      >>>         "param1": param1, 
      >>>         "param2": param2, 
      >>>         "param3": param3, 
      >>>         ..., # record other parameters
      >>>         "result": func(param1, param2, param3, ...) # save the calculation results
      >>>     }
      >>>     # return the dict for saved parameters and calculation results
      >>>     return res 

    The input of `func_helper` can only be set to the parameters that need to be adjusted and tested, 
    which can reduce the number of fixed parameters to be passed to `func`. \n

    Then, create a parameter list `param_list` to be passed to `func`(i.e. `func_helper`), 
    and then call `multi_task` to get the calculation results.

      >>> result = multi_task(func_help, param_list, n_job=4, verbose=1)
    
    Examples
    ----------
    Here is an example of multiplying two numbers: \n

      >>> def multiplication(a, b):
      >>>     return a * b
      >>> def mutiplication_helper(a, b): # auxiliary function for tracking parameters
      >>>     res = {
      >>>         "a": a, # track the first parameter
      >>>         "b": b, # track the second parameter
      >>>         "res": multiplication(a, b) # save result
      >>>     }
      >>>     return res
      >>> # create a parameter list, which consists of a dictionary with parameter signature
      >>> param_list = [{"a": a, "b": b} for a in range(2) for b in range(2)]
      >>> # call the multi process auxiliary function to get the calculation results
      >>> res = multi_task(mutiplication_helper, param_list, n_job = 4, verbose = 1)

    Print the calculation results. \n
    The output of the multi process auxiliary function is a list composed of dictionaries. \n
    Each dictionary stores the parameters used in the calculation and the calculation results.\n

      >>> for r in res:
      >>>     print("Param1: %s, Param2: %s, Result: %s"%(r["a"], r["b"], r["res"]))
      Param1: 0, Param2: 0, Result: 0
      Param1: 0, Param2: 1, Result: 0
      Param1: 1, Param2: 0, Result: 0
      Param1: 1, Param2: 1, Result: 1
    
    The results of multi process calculation can be used for subsequent analysis, such as finding the optimal parameters.
    """
    return Parallel(n_jobs=n_job, verbose=verbose)(delayed(func)(**param) for param in param_list)


# cross validation
def cross_valid(model, transfer : bool,
                X : np.ndarray, y : np.ndarray,
                metric : Optional[str]=None,
                folds : Optional[int]=4,
                parallel : Optional[bool]=False,
                n_jobs : Optional[int]=None,
                random_state : Optional[int]=0,
                *args, **kwargs) -> None:
    """
    Do cross validation to find optimal params.\n

    Parameters
    ----------
    model : Any
        The model instance to be trained, the `fit()` method should be provided.
    transfer : bool
        Whether do transfer learning.
    X : np.ndarray
        The observations.
    y : np.ndarray
        The labels.
    metric : str, default = `None`
        Specify evaluation indicators.
    folds : int, default = `4`
        The number of folds for cross validation.
    n_jobs : int, default = `None`
        The number of parallel processes.
    random_state : int, default = `0`
        Random seed used for cross validation.
    
    Return
    ----------
    score : float
        Return cross validation evaluation score
    """
    # model training and calculate metric
    def metrics(model : float, transfer : bool,
                X_train : np.ndarray, y_train : np.ndarray,
                X_valid : np.ndarray, y_valid : np.ndarray,
                metric : Optional[str]=None):
        model.fit(X=X_train,y=y_train,warm_up=False,transfer=transfer)
        return model.score(X_valid,y_valid,metric)
    # cross validation spliter
    if model.task == "regression":
        spliter = KFold(n_splits=folds,shuffle=True,random_state=random_state)
    else:
        spliter = StratifiedKFold(n_splits=folds,shuffle=True,random_state=random_state)
    if parallel:
        # create parameters list
        param_list = []
        for train_idx, valid_idx in spliter.split(X,y):
            params = {
                "model": deepcopy(model),
                "transfer": transfer,
                "X_train": X[train_idx],
                "y_train": y[train_idx],
                "X_valid": X[valid_idx],
                "y_valid": y[valid_idx],
                "metric": metric
            }
            param_list.append(params)
        # use multi process acceleration
        n_jobs = folds if n_jobs is None else n_jobs
        scores = multi_task(func=metrics,param_list=param_list,n_job=n_jobs,verbose=0)
    else:
        # save scores
        scores = []
        for train_idx, valid_idx in spliter.split(X,y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[valid_idx], y[valid_idx]
            score = metrics(model,transfer,X_train,y_train,X_valid,y_valid,metric)
            scores.append(score)
    return np.mean(scores)

# using multiple processes for parallel cross validation to speed up
def cross_valid_parallel(models : list,
                         params : list,
                         X : np.ndarray, y : np.ndarray,
                         transfer : bool,
                         metric : Optional[str]=None,
                         folds : Optional[int]=4,
                         sub_parallel : Optional[bool]=False,
                         n_jobs : Optional[int]=10,
                         random_state : Optional[int]=0,
                         greater_is_better : Optional[bool]=False):
    """
    Using multiple processes for parallel cross validation to speed up.\n
    See also `cross_valid()`

    Parameters
    ----------
    models : list
        The model instances to be trained, the `fit()` method should be provided for each model in `models`.
    params : list
        Parameter space for cross validation.
    X : np.ndarray
        The observations.
    y : np.ndarray
        The labels.
    transfer : bool
        Whether do transfer learning.
    metric : str, default = `None`
        Specify evaluation indicators.
    folds : int, default = `4`
        The number of folds for cross validation.
    sub_parallel : bool, default = `False`
        Whether sub processes are parallel.
    n_jobs : int, default = `10`
        The number of parallel processes.
    random_state : int, default = `0`
        Random seed used for cross validation.
    greater_is_better : bool, default = `False`
        Determine whether the optimal objective function is maximized or minimized.
    
    Return
    ----------
    best_param : Any
        Return the best params.
    """
    def metrics(model, param):
        res = {
            "param": param,
            "score": cross_valid(model,transfer,X,y,metric,folds,sub_parallel,n_jobs,random_state)
        }
        return res
    
    # models and params should have the same number
    try:
        assert(len(models) == len(params))
    except:
        raise ValueError("models and params should have the same number of elements!")
    
    # generate params dict
    param_list = []
    for model,param in zip(models,params):
        params = {
            "model": deepcopy(model),
            "param": param
        }
        param_list.append(params)
    # use multi process acceleration
    scores = multi_task(func=metrics,param_list=param_list,n_job=n_jobs,verbose=0)
    scores = pd.DataFrame(scores)
    # find the best params
    scores = scores.sort_values(by="score",ascending=greater_is_better)
    best_param = scores.iloc[-1,0]
    
    return best_param