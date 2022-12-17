from typing import Optional, Union
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.base import BaseEstimator
from abc import abstractmethod
from tqdm import tqdm
from .utils import estimate_Lipschitz, BIC, cross_valid, batch_mat_prod
from .utils import linear_nuclear_solver, linear_lasso_solver

class MatRegBase:
    """
    Base class for matrix regression method
    """
    def __init__(self, _lambda : Optional[float]=1.0, _delta : Union[float, str, None]='auto', 
                 penalty : Optional[str]="nuclear", loss : Optional[str]="square-error",
                 max_steps : Optional[int]=100, max_steps_epoch : Optional[int]=20, eps : Optional[float]=1e-8) -> None:
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
               criterion : Optional[str]="BIC",
               bound : Optional[list[float]]=[0,30], 
               step : Optional[float]=1,
               grid : Optional[list]=None,
               folds : Optional[int]=5,
               tau : Optional[float]=1.0, 
               show : Optional[bool]=False, 
               *args, **kwargs) -> None:
        """
        tuning(self, X : np.ndarray, y : np.ndarray, 
               transfer : Optional[bool]=False,
               criterion : Optional[str]="BIC",
               bound : Optional[list[float]]=[0,30], 
               step : Optional[float]=1,
               grid : Optional[list]=None,
               folds : Optional[int]=5,
               tau : Optional[float]=1.0, 
               show : Optional[bool]=False, 
               *args, **kwargs) -> None
            fit the model and use specified criterion to select optimal parameters
        
        Parameters
        ----------
        X : np.ndarray
            Observations of covariate with shape `(B,P,Q)`
        y : np.ndarray
            Observations of label with length `B`
        transfer : bool, optional
            Whether to use transfer learning to estimate `coef`\n
            Please call `set_transfer_coef()` first.
        criterion : str, optional
            Criterion to select optimal parameters, default is `"BIC"`\n
            Classification task only supports `"cv"`
        bound : list of float, optional
            Upper bound and lower bound of the search grid for `lambda`, default is `[0, 30]`
        step : float, optional
            The search grid step, default is `1`
        grid : list, optional
            The search grid of `lambda`, default is `None`\n
            If set, `bound` and `step` will be ignored
        folds : int, optional
            The number of folds for cross validation criterion, default is `5`
        tau : float, optional
            The penalty parameter used in BIC estimation procedure when do `Ridge()`, default is `1.0`
        show : bool, optional
            Whether to plot the criterion curve, default is `False`        
        """
        # init search grid
        if grid is None:
            grid = np.arange(bound[0],bound[1],step)
        
        # init
        self._historyCriterion = {}

        # dimension
        N, P, Q = X.shape[0], X.shape[1], X.shape[2]
        
        if self.task == "regression":
            print("Estimate variance of noise...")
            # Ridge estimate
            ridge = Ridge(alpha=tau,fit_intercept=False,max_iter=10000)
            ridge.fit(X.reshape(N,P*Q),y)
            singular_lse = self.__SVD(ridge.coef_.reshape(P,Q))

            # variance estimate
            lasso = Lasso(alpha=0.01,fit_intercept=False,max_iter=10000) # Lasso(alpha=1,fit_intercept=False)
            # lasso = LassoCV(cv=3,fit_intercept=False)
            lasso.fit(X.reshape(N,P*Q),y)
            err = lasso.predict(X.reshape(N,P*Q)) - y
            var = np.var(err)

            print("The estimated variance of noise is %.2f"%(var))
        elif self.task == "classification":
            # only support cross validation criterion
            criterion = "cv"

            # number of classes
            if self.n_class is None:
                self.classes = np.int_(np.unique(y))
                self.n_class = len(self.classes)

            # # Logistic Ridge
            # lr_ridge = LogisticRegression(penalty="l2",C=1/tau,fit_intercept=False)
            # lr_ridge.fit(X.reshape(N,P*Q),y)
            # singular_lse = self.__SVD(lr_ridge.coef_.reshape(self.n_class,P,Q))

            # # variace estimate
            # lr_lasso = LogisticRegression(penalty="l1",C=0.1,fit_intercept=False,solver="liblinear")
            # # lr_lasso = LogisticRegressionCV(penalty="l1",cv=3,fit_intercept=False,solver="liblinear",verbose=50,max_iter=100)
            # lr_lasso.fit(X.reshape(N,P*Q),y)
            # err = lr_lasso.predict(X.reshape(N,P*Q)) - y
            # var = np.var(err)
        
        # show traning config
        print("Using %s as criterion for %s task"%(criterion,self.task))

        # Setting initial delta one or two orders of magnitude larger than 1 / L
        print("Estimate Lipschitz constant...")
        delta = (1 / estimate_Lipschitz(X,self.task)) * 100

        # init best result
        optim_lambda, optim_coef, optim_score = None, None, None

        # init process bar
        pbar = tqdm(total=len(grid),ncols=120)

        # do grid search
        for i,_lambda in enumerate(grid):
            # update progress
            pbar.update(1)
            # show progress
            if self.task == "classification" and self.n_class > 2:
                if i == 0:
                    rank = "unknown"
                else:
                    rank = 0
                    for c in self.classes:
                        rank += np.sum(self.singular_vals[c] > 0)
                    rank /= self.n_class
            else:
                rank = np.sum(self.singular_vals > 0) if i > 0 else "unknown"
            
            pbar.set_description("Training for lambda = %.2f, Rank of the estimated coef : %s"%(_lambda, rank))

            # set lambda and delta
            self._lambda = _lambda
            self._delta = delta

            # use BIC
            if criterion == "BIC":
                # optimization
                self.fit(X,y,False,transfer,*args,**kwargs)

                # compute BIC
                y_pred = self.predict(X)
                lambda_score = BIC(y,y_pred,var,_lambda,self.singular_vals,singular_lse,tau,P,Q)

            # use cross-validation
            elif criterion == "cv":
                # cimpute cv score
                lambda_score = cross_valid(X,y,folds,self)

            # add history
            self._historyCriterion[_lambda] = lambda_score

            # update optim solution
            if optim_score is None or lambda_score < optim_score:
                optim_score = lambda_score
                optim_coef = self.coef_
                optim_lambda = self._lambda
        
        pbar.clear()
        pbar.close()

        # set best solution
        self.coef_ = optim_coef
        self._lambda = optim_lambda

        print("Best penalty lambda is %.4f"%(self._lambda))

        # for cross validation, we need train the model using full datasets
        if criterion == "cv":
            print("Training model with best parammeter using full datasets...")
            self._delta = delta
            self.fit(X,y,False,transfer,*args,**kwargs)

        # plot criterion curve
        if show:
            fig = plt.figure(figsize=(6,4),dpi=80)
            plt.plot(
                self._historyCriterion.keys(),
                self._historyCriterion.values(),
                c='b',ls='--'
            )
            plt.xlabel("$\lambda$")
            plt.ylabel("Criterion")
            plt.title("Criterion Curve")

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

        if transfer:
            try:
                assert(self.transfer_coef is not None)
            except:
                raise AttributeError("The auxiliary coefficient of transfer learning has not been set, please call set_transfer_coef() first!")
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
            self.coef_pre = np.zeros((p,q)) # np.random.randn(p,q) * 0.1
            self.coef_ = self.coef_pre.copy()

        # estimate delta  
        if self._delta is None:
            #  Setting initial delta one or two orders of magnitude larger than 1 / L
            self._delta = (1 / estimate_Lipschitz(X,self.task)) * 1000

        if self.penalty_ == "nuclear":
            historyObj, singular_vals, step = \
                linear_nuclear_solver(self,X,y,self._delta,self.max_steps,self.max_steps_epoch,self.eps)
            self._historyObj = historyObj.copy()
            self.singular_vals = singular_vals.copy()
            self._step = step
        elif self.penalty_ == "lasso":
            linear_lasso_solver(self,X,y,self.max_steps,transfer=self.transfer)
    
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
