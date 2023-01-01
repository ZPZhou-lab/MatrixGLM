# function used to generate simulated coefficients

import numpy as np
import sys

def GenerateCoef_vec(R : int = 5, s : float=0.50, p1 : int=64, p2 : int=64, h : float=0, K : int=10):
    """
    Under A^{vec} setting and A is known, generate the target domain and source domain coefficient.

    Parameters
    ----------
    R : int, default = `5`
        The rank of the matrix coef.
    s : float, default = `0.5`
        The sparsity of the matrix coef.
    p1, p2 : int, default = `64`
        THe height and width of the coef.
    h : float, default = `0`
        The deviation level of source domain and target domain.
    K : int, defualt = `10`
        THe number of the imformative source domain.

    Return
    ----------
    B : np.ndarray
        The true coef of target domain with shape `(p1,p2)`
    W : list
        The list of source domians' coef
    """
    # generate true coef for target domain
    p = np.sqrt(1 - (1-s)**(1/R))
    B1 = np.random.binomial(1,p,size=(p1,R))
    B2 = np.random.binomial(1,p,size=(p2,R))
    E = np.identity(R)
    B = B1 @ E @ B2.T
    # generate source domian coef
    W = []
    for k in range(K):
        # D_{i,j} in {-1,1} randomly with equal probability
        D = np.random.binomial(1,0.5,size=(p1,p2)) * 2 - 1
        Wk = B + (h/(p1*p2)) * D
        W.append(Wk.copy())
    return B, W


def GenerateCoef_sigma(R : int = 5, s : float=0.50, p1 : int=64, p2 : int=64, h : float=0, K : int=10):
    """
    Under A^{sigma} setting and A is known, generate the target domain and source domain coefficient.

    Parameters
    ----------
    R : int, default = `5`
        The rank of the matrix coef.
    s : float, default = `0.5`
        The sparsity of the matrix coef.
    p1, p2 : int, default = `64`
        THe height and width of the coef.
    h : float, default = `0`
        The deviation level of source domain and target domain.
    K : int, defualt = `10`
        THe number of the imformative source domain.

    Return
    ----------
    B : np.ndarray
        The true coef of target domain with shape `(p1,p2)`
    W : list
        The list of source domians' coef
    """
    p = np.sqrt(1 - (1-s)**(1/R))
    B1 = np.random.binomial(1,p,size=(p1,R))
    B2 = np.random.binomial(1,p,size=(p2,R))
    E = np.identity(R)
    B = B1 @ E @ B2.T
    # generate source domian coef
    W = []
    for k in range(K):
        err = np.random.rand(R)
        err = err / np.linalg.norm(err)
        err = err * (h / (R*np.sqrt(min(p1,p2))))
        D = E + np.diag(err)
        Wk = B1 @ D @ B2.T
        W.append(Wk.copy())
    return B, W

def GenerateCoef_vec_unknown(
    R : int = 5, s : float=0.50, p1 : int=64, p2 : int=64, h : float=0, 
    K : int=10, Ka : int=10, sk : float=0.50):
    """
    Under A^{vec} setting and A is known, generate the target domain and source domain coefficient.

    Parameters
    ----------
    R : int, default = `5`
        The rank of the matrix coef.
    s : float, default = `0.5`
        The sparsity of the matrix coef.
    p1, p2 : int, default = `64`
        THe height and width of the coef.
    h : float, default = `0`
        The deviation level of source domain and target domain.
    K : int, defualt = `10`
        THe number of the source domain.
    Ka : int, defualt = `10`
        THe number of the imformative source domain.

    Return
    ----------
    B : np.ndarray
        The true coef of target domain with shape `(p1,p2)`
    W : list
        The list of source domians' coef
    """
    # generate true coef for target domain
    p = np.sqrt(1 - (1-s)**(1/R))
    B1 = np.random.binomial(1,p,size=(p1,R))
    B2 = np.random.binomial(1,p,size=(p2,R))
    E = np.identity(R)
    B = B1 @ E @ B2.T
    # generate imformative source domian coef
    W = []
    for k in range(Ka):
        # D_{i,j} in {-1,1} randomly with equal probability
        D = np.random.binomial(1,0.5,size=(p1,p2)) * 2 - 1
        Wk = B + (h/(p1*p2)) * D
        W.append(Wk.copy())
    # generate other source domain coef
    Ix, Iy = np.where(B == 0) # the index where B == 0
    n = len(Ix)
    m = int(n*sk)
    for k in range(K-Ka):
        sampleds = np.random.choice(n,m,False)
        Ikx, Iky = Ix[sampleds], Iy[sampleds]
        Wk = np.random.binomial(1,0.5,size=(p1,p2)) * 2 - 1
        Wk = (h/(p1*p2)) * Wk
        Wk[Ikx,Iky] += 1
        W.append(Wk.copy())
    return B, W

def GenerateCoef_sigma_unknown(
    R : int = 5, s : float=0.50, p1 : int=64, p2 : int=64, h : float=0, 
    K : int=10, Ka : int=10, sk : float=0.50):
    """
    Under A^{vec} setting and A is known, generate the target domain and source domain coefficient.

    Parameters
    ----------
    R : int, default = `5`
        The rank of the matrix coef.
    s : float, default = `0.5`
        The sparsity of the matrix coef.
    p1, p2 : int, default = `64`
        THe height and width of the coef.
    h : float, default = `0`
        The deviation level of source domain and target domain.
    K : int, defualt = `10`
        THe number of the source domain.
    Ka : int, defualt = `10`
        THe number of the imformative source domain.

    Return
    ----------
    B : np.ndarray
        The true coef of target domain with shape `(p1,p2)`
    W : list
        The list of source domians' coef
    """
    # generate true coef for target domain
    p = np.sqrt(1 - (1-s)**(1/R))
    B1 = np.random.binomial(1,p,size=(p1,R))
    B2 = np.random.binomial(1,p,size=(p2,R))
    E = np.identity(R)
    B = B1 @ E @ B2.T
    # generate imformative source domian coef
    W = []
    for k in range(Ka):
        err = np.random.rand(R)
        err = err / np.linalg.norm(err)
        err = err * (h / (R*np.sqrt(min(p1,p2))))
        D = E + np.diag(err)
        Wk = B1 @ D @ B2.T
        W.append(Wk.copy())
    # generate other source domain coef
    Ix, Iy = np.where(B == 0) # the index where B == 0
    n = len(Ix)
    m = int(n*sk)
    for k in range(K-Ka):
        sampleds = np.random.choice(n,m,False)
        Ikx, Iky = Ix[sampleds], Iy[sampleds]
        Wk = np.random.binomial(1,0.5,size=(p1,p2)) * 2 - 1
        Wk = (h/(p1*p2)) * Wk
        Wk[Ikx,Iky] += 1
        W.append(Wk.copy())
    return B, W