from typing import Optional, Union
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from .MatrixGLMBase import MatrixGLMBase
from .utils import *
from . import solver

# Classifier for classification problem
class MatrixClassifier(MatrixGLMBase):
    # init constructor
    def __init__(self, 
                _lambda: Optional[float] = 1, 
                _delta: Union[float, str, None] = 'auto', 
                penalty: Optional[str] = "nuclear", 
                loss: Optional[str] = "neg-loglike", 
                multi_class : Optional[str] = "auto",
                max_steps: Optional[int] = 100, 
                max_steps_epoch: Optional[int] = 20, 
                eps: Optional[float] = 1e-8) -> None:
        """
        MatClassifier(self,                 
                      _lambda: Optional[float] = 1, 
                      _delta: Union[float, str, None] = 'auto', 
                      penalty: Optional[str] = "nuclear", 
                      loss: Optional[str] = "neg-loglike", 
                      multi_class : Optional[str] = "auto",
                      max_steps: Optional[int] = 50, 
                      max_steps_epoch: Optional[int] = 20, 
                      eps: Optional[float] = 1e-8) -> None) -> None
            Nesterov Matrix Claasifier
        
        Parameters
        ----------
        _lambda : float, optional
            Penalty parameter, default is `1.0`
        _delta : float, str, optional
            Provided Lipschitz constant `1 / L`, default is `'auto'`.\n
            When set to `'auto'`, it will use Fisher Imformation to estimate Lipschitz constant `L` 
            and set initial `delta` one or two orders of magnitude larger than `1 / L`
        loss : str, optional
            Loss function, default is `'neg-loglike'`
        multi_class : str, optional
            THe method for multi-class problem, default is `auto"`, can be set as `{"auto", "binary", "multinomial"}`\n
            when `n_class > 2`, the model will use `'multinomial'` method to estimate `coef` and obtain `n_class` group coefs\n
            wHen `n_class = 2`, the model will use `'binary'` as default and obtain one group coef,\n 
            set to `'multinomial'` will obtain two group coefs
    
        penalty : str, optional
            Penalty method, default is `'nuclear'`
            `J(B) = \\lambda\\cdot\\|\B\|_*`
        max_steps : int, optional
            The max iteration steps for model optimization, default is `50`
        max_steps_epoch : int, optional
            The max iteration steps for each epoch during Armijo line search, default is `25`
        eps : float, optional
            Convergence threshold of objective function, default is `1e-8`
        """
        super().__init__(_lambda, _delta, penalty, loss, max_steps, max_steps_epoch, eps)

        # classification threshold
        self.threshold = 0.5
        self.task = 'classification'

        # number of classes
        self.n_class = None
        self.classes = None
        # method for multi-class
        self.multi_class = multi_class
        # logits for transfer learning
        self.transfer_logits = None

    # predict
    def predict(self, X : np.ndarray) -> np.ndarray:
        # dimension check
        assert(len(X.shape) == 3)
        if self.multi_class == "binary":
            p, q = self.coef_.shape[0], self.coef_.shape[1]
        else:
            p, q = self.coef_[0].shape[0], self.coef_[0].shape[1]
        assert(X.shape[1] == p and X.shape[2] == q)

        if self.multi_class == "binary":
            logits = batch_mat_prod(X,self.coef_)
            return np.int_(sigmoid(logits) >= self.threshold)
        else:
            proba = self.predict_proba(X)
            return np.argmax(proba,axis=1)
    
    # predict probability
    def predict_proba(self, X : np.ndarray) -> np.ndarray:
        """
        predict_proba(self, X : np.ndarray) -> np.ndarray
            predict classification probability

        Parameters
        ----------
        X : np.ndarray
            Observations with shape `(B,P,Q)`

        Return
        ----------
        proba : np.ndarray
            The probabilities with shape `(B,2)`
        """
        # dimension check
        assert(len(X.shape) == 3)
        if self.multi_class == "binary":
            p, q = self.coef_.shape[0], self.coef_.shape[1]
        else:
            p, q = self.coef_[0].shape[0], self.coef_[0].shape[1]
        assert(X.shape[1] == p and X.shape[2] == q)

        if self.multi_class == "binary":
            logits = batch_mat_prod(X,self.coef_)
            sigmoid_logits = sigmoid(logits).reshape(-1,1)
            proba = np.hstack((1-sigmoid_logits,sigmoid_logits))
        else:
            # compute logits
            logits = Logits(X,self.coef_)
            # compute proba
            proba = softmax(logits)

        return proba
    
    # evaluation
    def score(self, X : np.ndarray, y : np.ndarray, metric : Optional[str]=None) -> float:
        metric = "accuracy" if metric == None else metric
        y_pred = self.predict(X)
        if metric in ["accuracy","acc","accuracy_score"]:
            s = accuracy_score(y,y_pred)
        elif metric in ["error_rate"]:
            s = 1 - accuracy_score(y,y_pred)
        elif metric in ["auc","roc_auc"]:
            if self.multi_class == "multinomial" and len(y.shape) == 1:
                y = OneHotEncoder().fit_transform(y)
            if self.multi_class == "binary":
                proba = self.predict_proba(X)[:,1]
            else:
                proba = self.predict_proba(X)
            s = roc_auc_score(y_true=y,y_score=proba)
        elif metric in ["f1_score"]:
            method = "binary" if self.multi_class == "binary" else "micro"
            s = f1_score(y,y_pred,average=method)
        elif metric == "loss":
            if self.transfer:
                logits = batch_mat_prod(X,self.coef_) + batch_mat_prod(X,self.transfer_coef)
            else:
                logits = batch_mat_prod(X,self.coef_)
            s = self.loss(X,y,self.coef_,logits=logits)
        else:
            raise ValueError("Not supported for metric: %s"%(metric))
        return s

    # rebuild for optimize
    def fit(self, X: np.ndarray, y: np.ndarray, warm_up: Optional[bool] = False, 
            transfer : Optional[bool]=False, *args, **kwargs) -> None:
        # dimension check
        assert(X.shape[0] == len(y))
        assert(len(X.shape) == 3)

        # if not worm up, init alpha, step, coef
        if not warm_up:
            # reset
            self._delta = None
            self.coef_ = None
            # init number of class
            self.n_class  = None

        # number of classes
        if self.n_class is None:
            self.classes = np.int_(np.unique(y))
            self.n_class = len(self.classes)
        
        # set multi-class method
        if self.multi_class == "auto":
            self.multi_class = "binary" if self.n_class == 2 else "multinomial"
        if self.n_class > 2:
            self.multi_class = "multinomial"

        # set tranfer config
        if transfer:
            try:
                assert(self.transfer_coef is not None)
            except:
                raise AttributeError("The auxiliary coefficient of transfer learning has not been set, please call set_coef() first!")
            self.transfer_logits = batch_mat_prod(X,self.transfer_coef) if self.multi_class == "binary" else Logits(X,self.transfer_coef)
            self.transfer = True
        else:
            self.transfer = False
        
        # The parent class method can be reused for binary classification problems
        if self.multi_class == "binary":
            return super().fit(X, y, warm_up, transfer, *args, **kwargs)
        # The logic of dealing with multiclass classification problems
        elif self.multi_class == "multinomial":
            # estimate delta
            if self._delta is None:
                #  Setting initial delta one or two orders of magnitude larger than 1 / L
                self._delta =  (1  / (self.n_class * estimate_Lipschitz(X, self.task))) * 1000
            
            # set dimension
            self.dim = X.shape[1:]
            p, q = X.shape[1], X.shape[2]

            # init beta
            if self.coef_ is None:
                self.coef_ = np.zeros((self.n_class,p,q)) # np.random.randn(self.n_class,p,q) * 0.1

            if self.penalty_ == "nuclear":
                historyObj, singular_vals, step = \
                    solver.multinomial_nuclear_solver(self,X,y,self._delta,self.max_steps,self.max_steps_epoch,self.eps)
                self._historyObj = historyObj.copy()
                self.singular_vals = singular_vals.copy()
                self._step = step
            elif self.penalty_ == "lasso":
                solver.multinomial_lasso_solver(self,X,y,self.max_steps,transfer=self.transfer)

    # compute penalty
    def penalty(self, coef: np.ndarray) -> float:
        # penalty for multinomial
        if self.multi_class == "multinomial":
            penalty = 0
            for c in range(coef.shape[0]):
                # nuclear-norm penalty
                if self.penalty_ == "nuclear":
                    _, singular_vals, _ = np.linalg.svd(coef[c],full_matrices=False)
                    penalty += self._lambda * np.abs(singular_vals).sum()
                # lasso-norm penalty
                elif self.penalty_ == "lasso":
                    penalty += self._lambda * np.abs(coef[c]).sum()
            return penalty / self.n_class
        return super().penalty(coef)
    
    # compute loss
    def loss(self, X : np.ndarray, y : np.ndarray, coef : np.ndarray, logits : Optional[np.ndarray]=None) -> float:
        # loss for multinomial
        if self.multi_class == "multinomial":
            # multinomial case only support neg-loglike
            # compute logits
            if logits is None:
                logits = Logits(X,coef) + self.transfer_logits if self.transfer else Logits(X,coef)
            
            if self.loss_ == "neg-loglike":
                return -1 * np.mean(np.sum(y*logits,axis=1) - np.log(np.sum(np.exp(logits),axis=1)))
        elif self.multi_class == "binary":
            # compute logits
            if logits is None:
                logits = batch_mat_prod(X,coef) + self.transfer_logits if self.transfer else batch_mat_prod(X,coef)
            # do sigmoid transform
            proba = sigmoid(logits)
            
            if self.loss_ == "neg-loglike":
                return -1 * np.mean(y * np.log(proba + 1e-16) + (1 - y) * np.log(1 - proba + 1e-16))

    # compute gradient of loss at point S
    def gradients(self, S : np.ndarray, X : np.ndarray, y : np.ndarray, logits : Optional[np.ndarray]=None) -> np.ndarray:
        # batch size
        B = X.shape[0]
        # loss for multinomial
        if self.multi_class == "multinomial":
            # compute logits
            if logits is None:
                logits = Logits(X,S) + self.transfer_logits if self.transfer else Logits(X,S)
            # shape : (B,n_class)
            proba = softmax(logits)

            if self.loss_ == "neg-loglike":
                return np.mean((proba - y).reshape((B,self.n_class,1,1)) * (np.expand_dims(X,1)),axis=0)
        elif self.multi_class == "binary":
            # compute logits
            if logits is None:
                logits = batch_mat_prod(X,S) + self.transfer_logits if self.transfer else batch_mat_prod(X,S)
            # do sigmoid transform
            proba = sigmoid(logits)
            
            if self.loss_ == "neg-loglike":
                grad = np.mean((proba - y).reshape((B,1,1)) * X,axis=0)
            return grad

    def approximate(self, coef: np.ndarray, S: np.ndarray, X: np.ndarray, y: np.ndarray, delta : float, Atmp : np.ndarray=None) -> float:
        if self.multi_class == "multinomial":
            loss = self.loss(X,y,S) # l(S)
            penalty = self.penalty(coef) # J(B)
            if Atmp is None:
                grad = self.gradients(S,X,y) # ∇l(S)
                inner_prod = grad.flatten() @ (coef - S).flatten() # <∇l(S), B-S>
                norm = inner_prod + np.sum(np.linalg.norm(coef - S,ord='fro',axis=(1,2)) ** 2) / (2*delta)
            else:
                norm = np.sum(np.linalg.norm(coef - Atmp,ord='fro',axis=(1,2)) ** 2) / (2*delta)
            return loss + norm + penalty

        return super().approximate(coef, S, X, y, delta, Atmp)
    
    