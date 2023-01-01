import numpy as np
from typing import List, Union, Optional, Callable
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from .MatrixClassifier import MatrixClassifier
from .MatrixRegressor import MatrixRegressor
from .MatrixGLMBase import MatrixGLMBase
from .utils import *

def transfer_model(model,_lambda,**kwargs):
    if model.task == "regression":
        return MatrixRegressor(_lambda=_lambda,penalty=model.penalty_transfer,max_steps=model.max_steps,**kwargs)
    elif model.task == "classification":
        return MatrixClassifier(_lambda=_lambda,penalty=model.penalty_transfer,max_steps=model.max_steps,**kwargs)

def debias_model(model,_lambda,**kwargs):
    if model.task == "regression":
        return MatrixRegressor(_lambda=_lambda,penalty=model.penalty_debias,max_steps=model.max_steps,**kwargs)
    elif model.task == "classification":
        return MatrixClassifier(_lambda=_lambda,penalty=model.penalty_debias,max_steps=model.max_steps,**kwargs)

class TransMatrixGLM(BaseEstimator):
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
        # number of oracle auxiliary datasets
        n_auxiliary = 0 if A is None else len(A)
        
        if n_auxiliary > 0:
            # transfer step
            model = self.transfer_step(Xt=Xt,yt=yt,Xa=Xa,ya=ya,A=A,**kwargs)
            omega = model.coef_.copy()

            # debias step
            model = self.debias_step(Xt=Xt,yt=yt,omega=omega,**kwargs)
            delta = model.coef_.copy()

            # compute true beta
            beta = omega + delta
            self.transfer_beta = omega
            self.debias_beta = delta
        # without auxiliary data
        else:
            model = self.debias_step(Xt=Xt,yt=yt,omega=None)
            beta = model.coef_.copy()

            self.transfer_beta = None
            self.debias_beta = beta

        # set parameters
        self.estimator.coef_ = beta.copy()
        if self.task == "classification":
            self.estimator.multi_class = model.multi_class
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
               lambda_max_ratio : Optional[float]=5,
               num_grids : Optional[int]=20,
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
        lambda_max_ratio : float, default = `5`
            Controal the `upper bound` of searched penalty coefficient,\n 
            the `upper bound` will set as `lambda_max_ratio * \sqrt{\log{p} / n}`.
        num_grids : int, default = `20`
            Control the number of grids between the lower and upper bound.
        """
        # check dimension
        assert(len(Xt.shape) == 3)
        N_target, p, q = Xt.shape[0], Xt.shape[1], Xt.shape[2]

        # number of oracle auxiliary datasets
        n_auxiliary = 0 if A is None else len(A)
    
        lambda_debias_base = np.sqrt((np.log(p*q))/N_target)
        lambda_debias_grid = np.hstack(
            [lambda_debias_base * np.array([1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,1]),
             lambda_debias_base * np.linspace(lambda_min_ratio,lambda_max_ratio,num_grids,endpoint=True)])
        
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
            lambda_transfer_grid = np.hstack(
                [lambda_transfer_base * np.array([1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,1]),
                 lambda_transfer_base * np.linspace(lambda_min_ratio,lambda_max_ratio,num_grids,endpoint=True)])
            transfer_models = []
            for lambda_transfer in lambda_transfer_grid:
                # build model
                model = transfer_model(self,lambda_transfer)
                transfer_models.append(model)
            # find best lambda
            best_lambda_transfer =\
                 cross_valid_parallel(models=transfer_models,params=lambda_transfer_grid,X=X_source,y=y_source,transfer=False,**kwargs)

            # build best model and training
            self.lambda_transfer = best_lambda_transfer
            model = self.transfer_step(Xt=Xt,yt=yt,Xa=Xa,ya=ya,A=A)
            # fetch coef
            omega = model.coef_.copy()

            """
            STEP 2: debias step
            """
            # do grid search using cross validation
            debias_models = []
            for lambda_debias in lambda_debias_grid:
                # build model
                model = debias_model(self,lambda_debias)
                model.set_transfer_coef(omega) # set tranfer coef
                debias_models.append(model)
            # find best lambda
            best_lambda_debias =\
                 cross_valid_parallel(models=debias_models,params=lambda_debias_grid,X=Xt,y=yt,transfer=True,**kwargs)

            self.lambda_debias = best_lambda_debias
            model = self.debias_step(Xt=Xt,yt=yt,omega=omega)
            # fetch coef
            delta = model.coef_.copy()

            # compute true beta
            beta = omega + delta
            self.transfer_beta = omega
            self.debias_beta = delta

        # without auxiliary data
        else:
            # do grid search using cross validation
            models = []
            for lambda_debias in lambda_debias_grid:
                # build model
                model = debias_model(self,lambda_debias)
                models.append(model)
            # find best lambda
            best_lambda_transfer = 0
            best_lambda_debias =\
                 cross_valid_parallel(models=models,params=lambda_debias_grid,X=Xt,y=yt,transfer=False,**kwargs)
            
            # build model using best params
            self.lambda_debias = best_lambda_debias
            model = self.debias_step(Xt=Xt,yt=yt,omega=None)

            beta = model.coef_.copy()
            self.transfer_beta = None
            self.debias_beta = beta
                
        # set parameters
        self.estimator.coef_ = beta.copy()
        if self.task == "classification":
            self.estimator.multi_class = model.multi_class
            self.estimator.classes = np.int_(np.unique(yt))
            self.estimator.n_class = len(self.estimator.classes)
        self.coef_ = beta.copy()

        return best_lambda_transfer, best_lambda_debias
    
    # fitting the model with unknown A
    def fit_unknown(self,
        Xt : np.ndarray, yt : np.ndarray,
        Xa : List[np.ndarray],
        ya : List[np.ndarray],
        m : Optional[int]=3, C0 : Optional[float]=2,
        *args, **kwargs) -> None:
        """
        Fit the model with unknown A and with the specified penalcy coefficient.\n

        Parameters
        ----------
        Xt : np.ndarray
            Observations from `target` domain
        yt : np.ndarray
            Labels from `target` domain
        Xa : list of np.ndarray
            Observation datasets from `auxiliary` domain.
        ya : list of np.ndarray
            Labels from `auxiliary` domain.
        m : int, default = `3`
            The number of folds for cross validation during detection.
        C0 : float, default = `2.0`
            The threshold for filtering source domain data.
        """
        # number of sources
        K = len(Xa)
        benchmark_metrics, metrics = np.zeros(m), np.zeros((m,K))
        # detection
        spliter = KFold(n_splits=m) if self.task == "regression" else StratifiedKFold(n_splits=m)
        for fold, (train_idx, valid_idx) in enumerate(spliter.split(X=Xt,y=yt)):
            Xt_train, yt_train = Xt[train_idx], yt[train_idx]
            Xt_valid, yt_valid = Xt[valid_idx], yt[valid_idx]
            # fit Lasso for benchmark
            benchmark_model = TransMatrixGLM(
                task=self.task,penalty_debias="lasso",lambda_debias=self.lambda_debias,
                max_steps=self.max_steps)
            benchmark_model.fit(Xt=Xt_train,yt=yt_train,A=None)
            metric = benchmark_model.score(Xt_valid,yt_valid)
            benchmark_metrics[fold] = metric

            # iterating on different domains
            for k in range(K):
                # do transfer step
                model = self.transfer_step(Xt=Xt_train,yt=yt_train,Xa=[Xa[k]],ya=ya[ya[0]],A=[0])
                metric = model.score(Xt_valid,yt_valid)
                metrics[fold,k] = metric
        
        # calculate the benchmark metric for lasso
        lasso_metric_mean = np.mean(benchmark_metrics)
        lasso_metric_std = np.std(benchmark_metrics)
        source_metirc_mean = np.mean(metrics,axis=0)
        # construct A set
        informative_A = np.where(source_metirc_mean - lasso_metric_mean <= C0*max(lasso_metric_std,0.01))[0]
        informative_A = list(informative_A)
        informative_A == None if informative_A == [] else informative_A

        # transfer learning
        self.fit(Xt=Xt,yt=yt,Xa=Xa,ya=ya,A=informative_A)

        return informative_A

    def tuning_unknown():
        ...
    
    # The transfer step for two-step Transfer Learning
    def transfer_step(self,
        Xt : np.ndarray, yt : np.ndarray,
        Xa : Optional[List[np.ndarray]] = None,
        ya : Optional[List[np.ndarray]] = None,
        A : Optional[list] = None,**kwargs) -> Union[MatrixRegressor,MatrixClassifier]:

        # check dimension
        assert(len(Xt.shape) == 3)
        assert(A is not None)
        N_target, p, q = Xt.shape[0], Xt.shape[1], Xt.shape[2]
        
        # concat auxiliary data
        X_source = np.vstack([Xa[i] for i in A] + [Xt])
        y_source = np.hstack([ya[i] for i in A] + [yt])
        N_source = len(X_source)
            
        # build model
        lambda_transfer = np.sqrt((np.log(p*q))/N_source) if self.lambda_transfer is None else self.lambda_transfer
        model = transfer_model(self,lambda_transfer,**kwargs)

        # training
        model.fit(X=X_source,y=y_source,warm_up=False,transfer=False)

        return model
    
    # The debias step for two-step Transfer Learning
    def debias_step(self,
        Xt : np.ndarray, yt : np.ndarray,
        omega : np.ndarray,**kwargs) -> Union[MatrixRegressor,MatrixClassifier]:
        N_target, p, q = Xt.shape[0], Xt.shape[1], Xt.shape[2]
        lambda_debias = np.sqrt((np.log(p*q))/N_target) if self.lambda_debias is None else self.lambda_debias
        # build model
        model = debias_model(self,lambda_debias,**kwargs)
        # training
        if omega is not None:
            model.set_transfer_coef(omega) # set tranfer coef
            model.fit(X=Xt,y=yt,warm_up=False,transfer=True)
        else:
            model.fit(X=Xt,y=yt,warm_up=False,transfer=False)

        return model
    
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