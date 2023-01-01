from typing import Optional, Union
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.base import BaseEstimator
from abc import abstractmethod
from tqdm import tqdm
from .utils import estimate_Lipschitz, BIC, cross_valid, batch_mat_prod
from . import solver

class MatrixGLMBase:
    """
    Base class for matrix GLM method
    """
    def __init__(self, _lambda : Optional[float]=1.0, _delta : Union[float, str, None]='auto', 
                 penalty : Optional[str]="nuclear", loss : Optional[str]="square-error",
                 max_steps : Optional[int]=100, max_steps_epoch : Optional[int]=20, eps : Optional[float]=1e-4) -> None:
        """
        NesterovMatReg(self, _lambda : Optional[float]=1.0, _delta : Union[float, str, None]='auto', 
                       penalty : Optional[str]="nuclear", loss : Optional[str]="square-error",
                       max_steps : Optional[int]=100, max_steps_epoch : Optional[int]=100, eps : Optional[float]=1e-8) -> None
            Nesterov Matrix Regression
        
        Parameters
        ----------
        _lambda : float, default = `1.0`
            Penalty parameter.
        _delta : float, str, default = `auto`
            Provided Lipschitz constant `1 / L`, default is `'auto'`.\n
            When set to `'auto'`, it will use Fisher Imformation to estimate Lipschitz constant `L` 
            and set initial `delta` one or two orders of magnitude larger than `1 / L`.
        loss : str, default = `'neg-loglike'`
            Loss function.
        penalty : str, default = `'nuclear'`
            Penalty method. can be one of `'nuclear'` and `'lasso'`.
        max_steps : int, default = `50`
            The max iteration steps for model optimization.
        max_steps_epoch : int, default = `25`
            The max iteration steps for each epoch during Armijo line search.
        eps : float, default = `1e-8`
            Convergence threshold of objective function.
        """
        self._step = 0 # global step

        # matrix coef init
        self.coef_pre = None
        self.coef_ = None
        # singular value of fitted coef
        self.singular_vals = None

        # delta to capture the unknown Lipschitz constant L
        if _delta == 'auto':
            self._delta = None
        else:
            self._delta = _delta
        # init for penalty weight
        self._lambda = _lambda
        # set loss function
        self.loss_ = loss
        # set penalty method
        self.penalty_ = penalty
        # init max steps
        self.max_steps = max_steps
        self.max_steps_epoch = max_steps_epoch # global max steps for Armijo line search

        # store objective history
        self._historyObj = []
        self.eps = eps

        # store criterion for optimization path
        self._historyCriterion = {}

        # dimensions
        self.dim = None
        # task
        self.task = None

        # fixed coef for transfer learning
        self.transfer = False
        self.transfer_coef = None
    
    # fit the model and use BIC to select optimal parameters
    def tuning(self, X : np.ndarray, y : np.ndarray, 
               transfer : Optional[bool]=False,
               lambda_min_ratio : Optional[float]=0.01,
               lambda_max_ratio : Optional[float]=1,
               num_grids : Optional[int]=10,
               folds : Optional[int]=5,
               metric : Optional[str]=None,
               show : Optional[bool]=False, 
               *args, **kwargs) -> None:
        """
        fit the model and use specified metric to select optimal parameters
        
        Parameters
        ----------
        X : np.ndarray
            Observations of covariate with shape `(B,P,Q)`
        y : np.ndarray
            Observations of label with length `B`
        transfer : bool, optional
            Whether to use transfer learning to estimate `coef`\n
            Please call `set_transfer_coef()` first.
        lambda_min_ratio : float, default = `0.01`
            Controal the `lower bound` of searched penalty coefficient,\n 
            the `lower bound` will set as `lambda_min_ratio * \sqrt{\log{p} / n}`.
        lambda_max_ratio : float, default = `1`
            Controal the `upper bound` of searched penalty coefficient,\n 
            the `upper bound` will set as `lambda_max_ratio * \sqrt{\log{p} / n}`.
        num_grids : int, default = `10`
            Control the number of grids between the lower and upper bound.
        folds : int, optional
            The number of folds for cross validation criterion, default is `5`
        metric : str, optional
            The specified metric to select optimal parameters, default is `None`.\n
            If `None`, use `mean_square_error()` for regression and `accuracy_score()` for classification.
        show : bool, optional
            Whether to plot the criterion curve, default is `False`        
        """
        # init search grid
        n, p, q = X.shape[0], X.shape[1], X.shape[2]
        grids = np.sqrt(np.log(p*q)/n) * np.linspace(lambda_min_ratio,lambda_max_ratio,num_grids,endpoint=True)
        
        # init
        historyCriterion = {}
        
        if self.task == "classification":
            # number of classes
            if self.n_class is None:
                self.classes = np.int_(np.unique(y))
                self.n_class = len(self.classes)

        # Setting initial delta one or two orders of magnitude larger than 1 / L
        delta = (1 / estimate_Lipschitz(X,self.task)) * 1000
        if self.task == "classification" and self.multi_class == "multinomial":
            delta /= self.n_class

        # init best result
        optim_lambda, optim_score = None, None

        for lambda_ in grids:
            self._lambda = lambda_
            self._delta = delta

            score = cross_valid(model=self,transfer=transfer,X=X,y=y,metric=metric,folds=folds,parallel=True)
            historyCriterion[lambda_] = score

            if optim_score is None or score < optim_score:
                optim_score = score
                optim_lambda = lambda_

        # plot criterion curve
        if show:
            fig = plt.figure(figsize=(6,4),dpi=80)
            plt.plot(
                historyCriterion.keys(),
                historyCriterion.values(),
                c='b',ls='--'
            )
            plt.xlabel("$\lambda$")
            plt.ylabel("Criterion")
            plt.title("Criterion Curve")
        
        return optim_lambda

    # optimization
    def fit(self, X : np.ndarray, y : np.ndarray, warm_up : Optional[bool]=False, 
            transfer : Optional[bool]=False, *args, **kwargs) -> None:
        """
        fit(self, X : np.ndarray, y : np.ndarray, warm_up : Optional[bool]=False, 
            transfer : Optional[bool]=False, *args, **kwargs) -> None
            fit the model to estimate underlying coef
        
        Parameters
        ----------
        X : np.ndarray
            Observations of covariate with shape `(B,P,Q)`
        y : np.ndarray
            Observations of label with length `B`
        warm_up : bool, optional
            Whether to update model parameters based on the trained model before. 
            Default is `False`
        transfer : bool, optional
            Whether to use transfer learning to estimate `coef`
        """
        # dimension check
        assert(X.shape[0] == len(y))
        assert(len(X.shape) == 3)

        # check whether transfer coef is provided
        if transfer:
            try:
                assert(self.transfer_coef is not None)
            except:
                raise AttributeError("The auxiliary coefficient of transfer learning has not been set, please call set_transfer_coef() first!")
            # for regression problem, we only need to fit the residuals
            if self.task == "regression":
                y = y - batch_mat_prod(X,self.transfer_coef)
            self.transfer = True
        else:
            self.transfer = False

        # set dimension
        self.dim = X.shape[1:]
        p, q = X.shape[1], X.shape[2]

        # if not worm up, init alpha, step, coef
        if not warm_up:
            # reset
            self._delta = None
            # init coef
            self.coef_ = np.zeros((p,q))

        # estimate delta  
        if self._delta is None:
            #  Setting initial delta one or two orders of magnitude larger than 1 / L
            self._delta = (1 / estimate_Lipschitz(X,self.task)) * 1000

        if self.penalty_ == "nuclear":
            historyObj, singular_vals, step = \
                solver.nuclear_solver(self,X,y,self._delta,self.max_steps,self.max_steps_epoch,self.eps)
            self._historyObj = historyObj.copy()
            self.singular_vals = singular_vals.copy()
            self._step = step
        elif self.penalty_ == "lasso":
            
            solver.lasso_solver(self,X,y,self.max_steps,transfer=self.transfer)
    
    # predict
    @abstractmethod
    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        predict(self, X : np.ndarray) -> np.ndarray
            predict response variables for given data

        Parameters
        ----------
        X : np.ndarray
            samples with shape `(B,P,Q)`
        
        Return
        ----------
        y : np.ndarray
            The prediction
        """
        ...
        raise NotImplementedError("predict method has not been defined yet!")

    # compute loss
    @abstractmethod
    def loss(self, X : np.ndarray, y : np.ndarray, coef : np.ndarray) -> float:
        """
        loss(self, X : np.ndarray, y : np.ndarray, coef : np.ndarray) -> float
            compute loss function for given samples and `coef`

        Parameters
        ----------
        X : np.ndarray
            samples with shape `(B,P,Q)`
        y : np.ndarray
            labels with shape `(B,)`
        coef : np.ndarray
            model coefficient with shape `(P,Q)`
        
        Return
        ----------
        loss : float
            The loss value of given samples and coef
        """
        ...
        raise NotImplementedError("loss function has not been defined yet!")

    # compute gradient of loss at point S
    @abstractmethod
    def gradients(self, S : np.ndarray, X : np.ndarray, y : np.ndarray) -> np.ndarray:
        """
        gradients(self, S : np.ndarray, X : np.ndarray, y : np.ndarray) -> np.ndarray
            compute gradient of loss at point S:\n
            `grad_{i,j} = B^{-1} * \sum_{b=1}^{B} {(y_b - <S, X_b>) * (-X_b_{i,j}) }`,\n
            in which, `i=1,...,P`, `j=1,...,Q`, `B` represents batch size.
        
        Parameters
        ----------
        S : np.ndarray
            iteration point generated by extrapolation
        X : np.ndarray
            samples with shape `(B,P,Q)`
        y : np.ndarray
            labels with shape `(B,)`
        
        Return
        ----------
        grad : np.ndarray
            The gradient of loss at point S
        """
        ...
        raise NotImplementedError("gradients of loss function has not been defined yet!")

    # compute penalty
    def penalty(self, coef : np.ndarray) -> float:
        penalty = 0
        if self.penalty_ == "nuclaer":
            _, singular_vals, _ = np.linalg.svd(coef,full_matrices=False)
            penalty = self._lambda * np.abs(singular_vals).sum()
        elif self.penalty_ == "lasso":
            penalty = self._lambda * np.abs(coef.flatten()).sum()
        return penalty

    # compute objective value
    def objective(self, X : np.ndarray, y : np.ndarray, coef : np.ndarray, 
                  loss : float=None, penalty : float=None) -> float:
        # compute penalty
        if penalty is None:
            penalty = self.penalty(coef)
        # compute loss
        if loss is None:
            loss = self.loss(X,y,coef)
        
        return loss + penalty
    
    # compute first order approximate
    def approximate(self, coef : np.ndarray, S : np.ndarray, 
                     X : np.ndarray, y : np.ndarray, delta : float, Atmp : np.ndarray=None) -> float:
        """
        approximate(self, coef : np.ndarray, S : np.ndarray, X : np.ndarray, y : np.ndarray) -> float
            compute first order approximate of ovjective function at point `S`
        
        Parameters
        ----------
        coef : np.ndarray
            Approximation value of objective function at point `coef`
        S : np.ndarray
            First order Taylor expansion at point `S`
        X : np.ndarray
            Observations of covariate `X`
        y : np.ndarray
            Observations of label `y`
        delta : float
            The inverse of the second-order Lopschitz constant.
        Atmp : np.ndarray
            The gradient descent point `S - \delta * ∇l(S)`
        
        Return
        ----------
        approx : float
            Approximation value of objective function at point `coef`
        """
        loss = self.loss(X,y,S) # l(S)
        penalty = self.penalty(coef) # J(B)
        if Atmp is None:
            grad = self.gradients(S,X,y) # ∇l(S)
            inner_prod = grad.flatten() @ (coef - S).flatten() # <∇l(S), B-S>
            norm = inner_prod + np.linalg.norm(coef - S,ord='fro') ** 2 / (2*delta)
        else:
            norm = np.linalg.norm(coef - Atmp,ord='fro') ** 2 / (2*delta)

        return loss + norm + penalty
    
    # show trainning process
    def history(self, *args, **kwargs) -> None:
        # plot the objective values
        fig = plt.figure(figsize=(4,4),dpi=80)
        plt.semilogy(self._historyObj,c='b',ls='--',*args,**kwargs)
        plt.xlabel("Steps")
        plt.ylabel("Objective Value")
    
    # show signal / image
    def show_signal(self, nrow : Optional[int]=None, ncol : Optional[int]=None, *args, **kwargs) -> None:
        if self.task == "regression" or (self.task == "classification" and self.multi_class == "binary"):
            fig = plt.figure(figsize=(4,4),dpi=80)
            plt.imshow(self.coef_,cmap=plt.cm.OrRd,
                       vmin=0,vmax=0.5*np.max(self.coef_))
            plt.title("Estimated Signal")
        else:
            # number of coef
            if nrow is None or ncol is None:
                n = self.coef_.shape[0]
                nrow = int(n**0.5)
                ncol = int(n / nrow) + 1
            fig,ax = plt.subplots(nrow,ncol,figsize=(3*ncol,3*nrow),dpi=60)
            ax = ax.flatten()
            for i,c in enumerate(self.classes):
                ax[i].imshow(self.coef_[i],cmap=plt.cm.OrRd,
                             vmin=0,vmax=0.5*np.max(self.coef_[i]))
                rank = np.linalg.matrix_rank(self.coef_[i])
                ax[i].set_title("class %s\nrank of estimated coef: %d"%(c,rank))
            plt.tight_layout()
            
    # set transfer leaning coefs
    def set_transfer_coef(self, coef : np.ndarray) -> None:
        self.transfer_coef = coef.copy()
    
    # evaluate
    def score(self, X : np.ndarray, y : np.ndarray, metric : Optional[str]=None) -> float:
        """
        score(self, X : np.ndarray, y : np.ndarray) -> float
            Compute the evaluation metrics on given data.
        
        Parameters
        ----------
        X : np.ndarray
            The given observations.
        y : np.ndarray
            The given labels
        metric : str, default = `None`
            The metric used for evaluation
        
        Return
        ----------
        score : float
            Score of the metric.
        """
        ...
        raise NotImplementedError("evaluation score has not been defined yet!")

    # SVD
    def __SVD(self, coef : np.ndarray) -> np.ndarray:
        """
        __SVD(self, coef : np.ndarray) -> np.ndarray
            Singular Value Decomposition
        
        Parameters
        ----------
        coef : np.ndarray
            Shape of `coef` could be `(P,Q)` or `(n_class,P,Q)`\n
        """
        if len(coef.shape) == 2:
            _, singular_vals, _ = np.linalg.svd(coef,full_matrices=False)
        elif len(coef.shape) == 3:
            singular_vals = np.zeros((coef.shape[0],min(coef.shape[1],coef.shape[2])))
            for c in range(coef.shape[0]):
                singular_vals[c] = self.__SVD(coef=coef[c])
        
        return singular_vals
