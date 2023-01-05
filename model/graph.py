import pandas as pd
import numpy as np

from .MatrixClassifier import MatrixClassifier
from .MatrixRegressor import MatrixRegressor
from .MatrixTransfer import TransMatrixGLM
from .evaluation import Graph

from pyriemann.estimation import Xdawn, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

from typing import Optional, Any

LAMBDA_MIN_RATIO = 0.01
LAMBDA_MAX_RATIO = 1.0
NUM_GRIDS = 10

# Training on target domain using Lasso only
class LassoOnly(Graph):
    def __init__(self, name: str, transfer : str, **kwargs) -> None:
        super().__init__(name,transfer)
        # Lasso Classifier
        self.estimator = TransMatrixGLM(
            task="classification",penalty_debias="lasso",**kwargs
        )
    
    def fit(self, X_target: np.ndarray, y_target: np.ndarray, 
            X_source: Optional[np.ndarray] = ..., y_source: Optional[np.ndarray] = ...,
            n_jobs : Optional[int]=1, random_state : Optional[int]=0):
        # use target data only
        self.estimator.tuning(
            Xt=X_target,yt=y_target,Xa=X_source,ya=y_source,A=None,
            lambda_min_ratio=LAMBDA_MIN_RATIO,lambda_max_ratio=LAMBDA_MAX_RATIO,num_grids=NUM_GRIDS,
            metric="roc_auc",n_jobs=n_jobs,random_state=random_state)
    
    def predict(self, X: np.ndarray):
        return self.estimator.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        return self.estimator.predict_proba(X)[:,1]

# Training on target domain using Nuclear only
class NuclearOnly(Graph):
    def __init__(self, name: str, transfer : str, **kwargs) -> None:
        super().__init__(name,transfer)
        # Nuclear Classifier
        self.estimator = TransMatrixGLM(
            task="classification",penalty_debias="nuclear",**kwargs
        )
    
    def fit(self, X_target: np.ndarray, y_target: np.ndarray, 
            X_source: Optional[np.ndarray] = ..., y_source: Optional[np.ndarray] = ...,
            n_jobs : Optional[int]=1, random_state : Optional[int]=0):
        # use target data only
        self.estimator.tuning(
            Xt=X_target,yt=y_target,Xa=X_source,ya=y_source,A=None,
            lambda_min_ratio=LAMBDA_MIN_RATIO,lambda_max_ratio=LAMBDA_MAX_RATIO,num_grids=NUM_GRIDS,
            metric="roc_auc",n_jobs=n_jobs,random_state=random_state)
    
    def predict(self, X: np.ndarray):
        return self.estimator.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        return self.estimator.predict_proba(X)[:,1]

# Training on target and source domain using naive combined method and Lasso classifier
class LassoNaive(Graph):
    def __init__(self, name: str, transfer : str, *args, **kwargs) -> None:
        super().__init__(name,transfer)
        # Lasso Classifier
        self.estimator = TransMatrixGLM(
            task="classification",penalty_debias="lasso",**kwargs
        )
    
    def fit(self, X_target: np.ndarray, y_target: np.ndarray, 
            X_source: Optional[np.ndarray] = ..., y_source: Optional[np.ndarray] = ...,
            n_jobs : Optional[int]=1, random_state : Optional[int]=0):
        # without transfer
        if len(X_source) > 0:
            X_target = np.concatenate([X_target,X_source])
            y_target = np.concatenate([y_target,y_source])
        
        self.estimator.tuning(
            Xt=X_target,yt=y_target,Xa=X_source,ya=y_source,A=None,
            lambda_min_ratio=LAMBDA_MIN_RATIO,lambda_max_ratio=LAMBDA_MAX_RATIO,num_grids=NUM_GRIDS,
            metric="roc_auc",n_jobs=n_jobs,random_state=random_state)
    
    def predict(self, X: np.ndarray):
        return self.estimator.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        return self.estimator.predict_proba(X)[:,1]

# Training on target and source domain using naive combined method and Nuclear classifier
class NuclearNaive(Graph):
    def __init__(self, name: str, transfer : str, *args, **kwargs) -> None:
        super().__init__(name,transfer)
        # Nuclear Classifier
        self.estimator = TransMatrixGLM(
            task="classification",penalty_debias="nuclear",**kwargs
        )
    
    def fit(self, X_target: np.ndarray, y_target: np.ndarray, 
            X_source: Optional[np.ndarray] = ..., y_source: Optional[np.ndarray] = ...,
            n_jobs : Optional[int]=1, random_state : Optional[int]=0):
        # without transfer
        if len(X_source) > 0:
            X_target = np.concatenate([X_target,X_source])
            y_target = np.concatenate([y_target,y_source])
        
        self.estimator.tuning(
            Xt=X_target,yt=y_target,Xa=X_source,ya=y_source,A=None,
            lambda_min_ratio=LAMBDA_MIN_RATIO,lambda_max_ratio=LAMBDA_MAX_RATIO,num_grids=NUM_GRIDS,
            metric="roc_auc",n_jobs=n_jobs,random_state=random_state)
    
    def predict(self, X: np.ndarray):
        return self.estimator.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        return self.estimator.predict_proba(X)[:,1]

# Transfer learing using Lasso + Lasso classifier
class LassoLassoTransfer(Graph):
    def __init__(self, name: str, transfer : str, *args, **kwargs) -> None:
        super().__init__(name,transfer)
        self.estimator = TransMatrixGLM(
            task="classification",penalty_debias="lasso",penalty_transfer="lasso",**kwargs
        )
    
    def fit(self, X_target: np.ndarray, y_target: np.ndarray, 
            X_source: Optional[np.ndarray] = ..., y_source: Optional[np.ndarray] = ...,
            n_jobs : Optional[int]=1, random_state : Optional[int]=0):
        self.estimator.tuning(
            Xt=X_target,yt=y_target,Xa=[X_source],ya=[y_source],A=[0],
            lambda_min_ratio=LAMBDA_MIN_RATIO,lambda_max_ratio=LAMBDA_MAX_RATIO,num_grids=NUM_GRIDS,
            metric="roc_auc",n_jobs=n_jobs,random_state=random_state)

    def predict(self, X: np.ndarray):
        return self.estimator.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        return self.estimator.predict_proba(X)[:,1]

# Transfer learing using Nuclear + Lasso classifier
class NuclearLassoTransfer(Graph):
    def __init__(self, name: str, transfer : str, *args, **kwargs) -> None:
        super().__init__(name,transfer)
        self.estimator = TransMatrixGLM(
            task="classification",penalty_debias="lasso",penalty_transfer="nuclear",**kwargs
        )
    
    def fit(self, X_target: np.ndarray, y_target: np.ndarray, 
            X_source: Optional[np.ndarray] = ..., y_source: Optional[np.ndarray] = ...,
            n_jobs : Optional[int]=1, random_state : Optional[int]=0):
        self.estimator.tuning(
            Xt=X_target,yt=y_target,Xa=[X_source],ya=[y_source],A=[0],
            lambda_min_ratio=LAMBDA_MIN_RATIO,lambda_max_ratio=LAMBDA_MAX_RATIO,num_grids=NUM_GRIDS,
            metric="roc_auc",n_jobs=n_jobs,random_state=random_state)

    def predict(self, X: np.ndarray):
        return self.estimator.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        return self.estimator.predict_proba(X)[:,1]

# Transfer learing using Nuclear + Nuclear classifier
class NuclearNuclearTransfer(Graph):
    def __init__(self, name: str, transfer : str, *args, **kwargs) -> None:
        super().__init__(name,transfer)
        self.estimator = TransMatrixGLM(
            task="classification",penalty_debias="nuclear",penalty_transfer="nuclear",**kwargs
        )
    
    def fit(self, X_target: np.ndarray, y_target: np.ndarray, 
            X_source: Optional[np.ndarray] = ..., y_source: Optional[np.ndarray] = ...,
            n_jobs : Optional[int]=1, random_state : Optional[int]=0):
        self.estimator.tuning(
            Xt=X_target,yt=y_target,Xa=[X_source],ya=[y_source],A=[0],
            lambda_min_ratio=LAMBDA_MIN_RATIO,lambda_max_ratio=LAMBDA_MAX_RATIO,num_grids=NUM_GRIDS,
            metric="roc_auc",n_jobs=n_jobs,random_state=random_state)

    def predict(self, X: np.ndarray):
        return self.estimator.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        return self.estimator.predict_proba(X)[:,1]