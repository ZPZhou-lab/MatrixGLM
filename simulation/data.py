# function used to generate simulated datasets

import numpy as np
import sys
from typing import Union

sys.path.append("D:\Code\MatrixGLM")
from model.utils import batch_mat_prod

# generate samples and labels for regression problems
def regression_data(B : np.ndarray, W : list, n0 : int=200, nk : int=100, scale : Union[str,float]='none'):
    """
    Generate linear regression dataset.

    Parameters
    ----------
    B : np.ndarray
        The true coef of target domain with shape `(p1,p2)`
    W : list
        The list of source domians' coef
    n0 : int, default = `200`
        The number of samples of target domain.
    nk : int, default = `100`
        The number of samples of each source domain.
    scale : str or float, default = `none`
        Whether to adjust the variance of `X`\n
        if `none`, `X ~ N(0,1)`.\n
        if `auto`, `X ~ N(0,1)` and then `X = X / \sqrt{p1*p2}`\n
        if float provided, `X ~ N(0,1)` and then `X = X / scale`.

    Return
    ----------
    (X_target, X_source, y_target, y_source) : tuple
    """

    K = len(W)
    p1, p2 = B.shape[0], B.shape[1]
    if isinstance(scale,str):
        scale = 1 if scale == 'none' else np.sqrt(p1*p2)
    
    X_target = np.random.randn(n0,p1,p2) / scale
    y_target = batch_mat_prod(X_target,B)
    X_source, y_source = [], []
    for k in range(K):
        X_s = np.random.randn(nk,p1,p2) / scale
        y_s = batch_mat_prod(X_s,W[k])
        X_source.append(X_s.copy())
        y_source.append(y_s.copy())
    return X_target, X_source, y_target, y_source


# generate samples and labels for binary logistic problems
def binary_logistic_data(B : np.ndarray, W : list, n0 : int=200, nk : int=100, scale : Union[str,float]='none'):
    """
    Generate binary logistic dataset.

    Parameters
    ----------
    B : np.ndarray
        The true coef of target domain with shape `(p1,p2)`
    W : list
        The list of source domians' coef
    n0 : int, default = `200`
        The number of samples of target domain.
    nk : int, default = `100`
        The number of samples of each source domain.
    
    Return
    ----------
    (X_target, X_source, y_target, y_source) : tuple
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    K = len(W)
    p1, p2 = B.shape[0], B.shape[1]
    if isinstance(scale,str):
        scale = 1 if scale == 'none' else np.sqrt(p1*p2)
    
    X_target = np.random.randn(n0,p1,p2) / scale
    logits = batch_mat_prod(X_target,B)
    y_target = np.random.binomial(1,sigmoid(logits),size=n0)
    X_source, y_source = [], []
    for k in range(K):
        X_s = np.random.randn(nk,p1,p2) / scale
        logits_t = batch_mat_prod(X_s,W[k])
        y_s = np.random.binomial(1,sigmoid(logits_t),size=nk)
        X_source.append(X_s.copy())
        y_source.append(y_s.copy())
    return X_target, X_source, y_target, y_source