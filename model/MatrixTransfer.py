import numpy as np
from typing import List, Union, Optional, Callable
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold, StratifiedKFold
from .MatrixClassifier import MatrixClassifier
from .MatrixRegressor import MatrixRegressor
from .MatrixRegBase import MatRegBase
from .utils import *

class OracleTransReg(BaseEstimator):
    def __init__(self, task : str, verbose : Optional[bool]=False, 
                 penalty_transfer : Optional[str]="nuclear", penalty_debias : Optional[str]="lasso",
                 lambda_transfer : Optional[float]=None, lambda_debias : Optional[float]=None,
                 max_steps : Optional[int]=100, **kwargs) -> None:
        """
        Parameters
        ----------
        task : str
            The task type, can be one of `'regression'` or `'classification'`.
        verbose : bool, default = `False`
            Whether to show the training imformation.
        penalty_transfer : str, default = `'nuclear'`
            The penalty function for `transfer` step.
        penalty_debias : str, default = `'lasso'`
            The penalty function for `debias` step.
        lambda_transfer : float, default = `None`
            The penalty coef for `transfer` step. If `None`, will set it as `\sqrt{\log{p} / n}`.
        lambda_debias : float, default = `None`
            The penalty coef for `debias` step. If `None`, will set it as `\sqrt{\log{p} / n}`.
        max_steps : int, default = `100`
            The maximum number of iterations of the algorithm.
        """
        self.task = task
        self.estimator = MatrixRegressor(**kwargs) if task == "regression" else MatrixClassifier(**kwargs)
        self.verbose = verbose
        self.transfer_beta = None
        self.debias_beta = None
        self.penalty_transfer = penalty_transfer
        self.penalty_debias = penalty_debias
        self.lambda_transfer = lambda_transfer
        self.lambda_debias = lambda_debias
        self.max_steps = max_steps

    def fit(self, 
            Xt : np.ndarray, yt : np.ndarray,
            Xa : Optional[List[np.ndarray]] = None,
            ya : Optional[List[np.ndarray]] = None,
            A : Optional[list] = None,
            *args, **kwargs) -> None:
        """
        Fit the model with the specified penalty coefficient.\n

        Parameters
        ----------
        Xt : np.ndarray
            Observations from `target` domain
        yt : np.ndarray
            Labels from `target` domain
        Xa : list of np.ndarray, default = `None`
            Observation datasets from `auxiliary` domain.
        ya : list of np.ndarray, default = `None`
            Labels from `auxiliary` domain.
        A : list, default = `None`
            The imformative oracle auxiliary datasets index.
        """

        # check dimension
        assert(len(Xt.shape) == 3)
        N_target, p, q = Xt.shape[0], Xt.shape[1], Xt.shape[2]

        # number of oracle auxiliary datasets
        n_auxiliary = 0 if A is None else len(A)
    
        lambda_debias = np.sqrt((np.log(p*q))/N_target) if self.lambda_debias is None else self.lambda_debias
        
        if n_auxiliary > 0:
            # concat auxiliary data
            X_source = np.vstack([Xa[i] for i in A] + [Xt])
            y_source = np.hstack([ya[i] for i in A] + [yt])
            N_source = len(X_source)
            
            """
            STEP 1: transfer step
            """
            # build model
            lambda_transfer = np.sqrt((np.log(p*q))/N_source) if self.lambda_transfer is None else self.lambda_transfer
            transfer_model = MatrixRegressor(_lambda=lambda_transfer,penalty=self.penalty_transfer,max_steps=self.max_steps,**kwargs) if self.task == "regression"\
                else MatrixClassifier(_lambda=lambda_transfer,penalty=self.penalty_transfer,max_steps=self.max_steps,**kwargs)

            # training
            transfer_model.fit(X=X_source,y=y_source,warm_up=False,transfer=False)
            # fetch coef
            omega = transfer_model.coef_.copy()

            """
            STEP 2: debias step
            """
            # build model
            debias_model = MatrixRegressor(_lambda=lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps,**kwargs) if self.task == "regression"\
                else MatrixClassifier(_lambda=lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps,**kwargs)

            # training
            debias_model.set_transfer_coef(omega) # set tranfer coef
            debias_model.fit(X=Xt,y=yt,warm_up=False,transfer=True)
            # fetch coef
            delta = debias_model.coef_.copy()

            # compute true beta
            beta = omega + delta
            self.transfer_beta = omega
            self.debias_beta = delta
            if self.task == "classification":
                self.estimator.multi_class = debias_model.multi_class
        # without auxiliary data
        else:
            # build model
            model = MatrixRegressor(_lambda=lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps,**kwargs) if self.task == "regression"\
                else MatrixClassifier(_lambda=lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps,**kwargs)
            
            # training
            model.fit(X=Xt,y=yt,*args)
            beta = model.coef_.copy()
            self.transfer_beta = None
            self.debias_beta = beta
            if self.task == "classification":
                self.estimator.multi_class = debias_model.multi_class
    
        # set parameters
        self.estimator.coef_ = beta.copy()
        if self.task == "classification":
            self.estimator.classes = np.int_(np.unique(yt))
            self.estimator.n_class = len(self.estimator.classes)
        self.coef_ = beta.copy()

        return self
    
    def tuning(self, 
               Xt : np.ndarray, yt : np.ndarray,
               Xa : Optional[List[np.ndarray]] = None,
               ya : Optional[List[np.ndarray]] = None,
               A : Optional[list] = None,
               lambda_min_ratio : Optional[float]=0.01,
               lambda_max_ratio : Optional[float]=1,
               num_grids : Optional[int]=10,
               *args, **kwargs) -> None:
        """
        Select the optimal penalty coefficient through cross validation, and then fit the model.

        Parameters
        ----------
        Xt : np.ndarray
            Observations from `target` domain
        yt : np.ndarray
            Labels from `target` domain
        Xa : list of np.ndarray, default = `None`
            Observation datasets from `auxiliary` domain.
        ya : list of np.ndarray, default = `None`
            Labels from `auxiliary` domain.
        A : list, default = `None`
            The imformative oracle auxiliary datasets index.
        lambda_min_ratio : float, default = `0.01`
            Controal the `lower bound` of searched penalty coefficient,\n 
            the `lower bound` will set as `lambda_min_ratio * \sqrt{\log{p} / n}`.
        lambda_max_ratio : float, default = `1`
            Controal the `upper bound` of searched penalty coefficient,\n 
            the `upper bound` will set as `lambda_max_ratio * \sqrt{\log{p} / n}`.
        num_grids : int, default = `10`
            Control the number of grids between the lower and upper bound.
        """
        # check dimension
        assert(len(Xt.shape) == 3)
        N_target, p, q = Xt.shape[0], Xt.shape[1], Xt.shape[2]

        # number of oracle auxiliary datasets
        n_auxiliary = 0 if A is None else len(A)
    
        lambda_debias_base = np.sqrt((np.log(p*q))/N_target)
        lambda_debias_grid = lambda_debias_base * np.linspace(
            lambda_min_ratio,lambda_max_ratio,num_grids,endpoint=True)
        
        if n_auxiliary > 0:
            # concat auxiliary data
            X_source = np.vstack([Xa[i] for i in A] + [Xt])
            y_source = np.hstack([ya[i] for i in A] + [yt])
            N_source = len(X_source)
            
            """
            STEP 1: transfer step
            """
            # do grid search using cross validation
            lambda_transfer_base = np.sqrt((np.log(p*q))/N_source)
            lambda_transfer_grid = lambda_transfer_base * np.linspace(
                lambda_min_ratio,lambda_max_ratio,num_grids,endpoint=True)
            transfer_models = []
            for lambda_transfer in lambda_transfer_grid:
                # build model
                transfer_model = MatrixRegressor(_lambda=lambda_transfer,penalty=self.penalty_transfer,max_steps=self.max_steps) if self.task == "regression"\
                    else MatrixClassifier(_lambda=lambda_transfer,penalty=self.penalty_transfer,max_steps=self.max_steps)
                transfer_models.append(transfer_model)
            # find best lambda
            best_lambda_transfer = cross_valid_parallel(
                models=transfer_models,params=lambda_transfer_grid,
                X=X_source,y=y_source,transfer=False,**kwargs)

            # build best model and training
            transfer_model = MatrixRegressor(_lambda=best_lambda_transfer,penalty=self.penalty_transfer,max_steps=self.max_steps) if self.task == "regression"\
                else MatrixClassifier(_lambda=best_lambda_transfer,penalty=self.penalty_transfer,max_steps=self.max_steps)
            transfer_model.fit(X=X_source,y=y_source,warm_up=False,transfer=False)
            # fetch coef
            omega = transfer_model.coef_.copy()

            """
            STEP 2: debias step
            """
            # do grid search using cross validation
            debias_models = []
            for lambda_debias in lambda_debias_grid:
                # build model
                debias_model = MatrixRegressor(_lambda=lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps) if self.task == "regression"\
                    else MatrixClassifier(_lambda=lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps)
                debias_model.set_transfer_coef(omega) # set tranfer coef
                debias_models.append(debias_model)
            # find best lambda
            best_lambda_debias = cross_valid_parallel(
                models=debias_models,params=lambda_debias_grid,
                X=Xt,y=yt,transfer=True,**kwargs)

            # build best model and training
            debias_model = MatrixRegressor(_lambda=best_lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps) if self.task == "regression"\
                else MatrixClassifier(_lambda=best_lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps)
            # training
            debias_model.set_transfer_coef(omega) # set tranfer coef
            debias_model.fit(X=Xt,y=yt,warm_up=False,transfer=True)
            # fetch coef
            delta = debias_model.coef_.copy()

            # compute true beta
            beta = omega + delta
            self.transfer_beta = omega
            self.debias_beta = delta
            if self.task == "classification":
                self.estimator.multi_class = debias_model.multi_class
        # without auxiliary data
        else:
            # do grid search using cross validation
            models = []
            for lambda_debias in lambda_debias_grid:
                # build model
                model = MatrixRegressor(_lambda=lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps) if self.task == "regression"\
                    else MatrixClassifier(_lambda=lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps)
                models.append(model)
            # find best lambda
            best_lambda_debias = cross_valid_parallel(
                models=models,params=lambda_debias_grid,
                X=Xt,y=yt,transfer=False,**kwargs)
            
            # build model using best params
            model = MatrixRegressor(_lambda=best_lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps) if self.task == "regression"\
                else MatrixClassifier(_lambda=best_lambda_debias,penalty=self.penalty_debias,max_steps=self.max_steps)
            
            # training
            model.fit(X=Xt,y=yt,warm_up=False,transfer=False)
            beta = model.coef_.copy()
            self.transfer_beta = None
            self.debias_beta = beta
            if self.task == "classification":
                self.estimator.multi_class = model.multi_class
    
        # set parameters
        self.estimator.coef_ = beta.copy()
        if self.task == "classification":
            self.estimator.classes = np.int_(np.unique(yt))
            self.estimator.n_class = len(self.estimator.classes)
        self.coef_ = beta.copy()

        return self

    def predict(self, X : np.ndarray):
        return self.estimator.predict(X)
    
    def predict_proba(self, X : np.ndarray):
        if isinstance(self.estimator, MatrixClassifier):
            return self.estimator.predict_proba(X)
        else:
            raise AttributeError("There is no predict_proba method for regression model!")

    def score(self,*args):
        return self.estimator.score(*args)
    
    def show_transfer_coef(self, nrow : Optional[int]=None, ncol : Optional[int]=None, 
                           rmin : float=1, rmax : float=1, title : Optional[str]=None, *args, **kwargs):
        """
        show transfer coefficients, includes `estimated` beta, `transfer` beta and `debias` beta.

        Parameters
        ----------
        nrow : int, default = `None`
            Controls the number of rows in a subplots.
        ncol : int, default = `None`
            Controls the number of columns in a subplots.
        rmin : float, default = `1`
            Control the coefficient of `vmin` in imshow() function, will set `vmin = rmin*vmin`.
        rmax : float, default = `1`
            Control the coefficient of `vmax` in imshow() function, will set `vmax = rmax*vmax`.
        title : str, default = `None`
            Set the title of the plot.
        """
        if self.task == "regression" or self.estimator.multi_class == "binary":
            if self.transfer_beta is None:
                self.estimator.show_signal()
            else:
                fig, ax = plt.subplots(1,3,figsize=(6,2))
                ax = ax.flatten()
                vmin, vmax = self.coef_.min(), self.coef_.max()
                ax[0].imshow(self.coef_,vmin=vmin,vmax=vmax,cmap=plt.cm.OrRd)
                ax[1].imshow(self.transfer_beta,vmin=vmin,vmax=vmax,cmap=plt.cm.OrRd)
                ax[2].imshow(self.debias_beta,vmin=vmin,vmax=vmax,cmap=plt.cm.OrRd)
                plt.tight_layout()
        else:
            if self.transfer_beta is None:
                self.estimator.show_signal(nrow,ncol,*args,**kwargs)
            else:
                fig, ax = plt.subplots(3,self.estimator.n_class,figsize=(2*self.estimator.n_class,6))
                ax = ax.flatten()
                for c in self.estimator.classes:
                    vmin = rmin*min(self.coef_[c].min(), self.transfer_beta[c].min(), self.debias_beta[c].min())
                    vmax = rmax*max(self.coef_[c].max(), self.transfer_beta[c].max(), self.debias_beta[c].max())
                    ax[c].imshow(self.coef_[c],vmin=vmin,vmax=vmax,cmap=plt.cm.OrRd)
                    ax[self.estimator.n_class+c].imshow(self.transfer_beta[c],vmin=vmin,vmax=vmax,cmap=plt.cm.OrRd)
                    ax[2*self.estimator.n_class+c].imshow(self.debias_beta[c],vmin=vmin,vmax=vmax,cmap=plt.cm.OrRd)
                    ax[c].set_title("Class: %d"%(c))
                    if c == 0:
                        ax[c].set_ylabel("Estimated Beta")
                        ax[self.estimator.n_class+c].set_ylabel("Transfer Beta")
                        ax[2*self.estimator.n_class+c].set_ylabel("Debias Beta")
                plt.title(title)
                plt.tight_layout()