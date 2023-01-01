from typing import Optional, Union
import numpy as np
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from .MatrixGLMBase import MatrixGLMBase
from .utils import batch_mat_prod
            
# Regressor for regression problem
class MatrixRegressor(MatrixGLMBase):
    # init constructor
    def __init__(self, 
                _lambda: Optional[float] = 1, 
                _delta: Union[float, str, None] = 'auto', 
                penalty: Optional[str] = "nuclear", 
                loss: Optional[str] = "neg-loglike", 
                max_steps: Optional[int] = 100, 
                max_steps_epoch: Optional[int] = 20, 
                eps: Optional[float] = 1e-8) -> None:
        """
        MatRegressor(self,                 
                     _lambda: Optional[float] = 1, 
                     _delta: Union[float, str, None] = 'auto', 
                     penalty: Optional[str] = "nuclear", 
                     loss: Optional[str] = "neg-loglike", 
                     max_steps: Optional[int] = 50, 
                     max_steps_epoch: Optional[int] = 20, 
                     eps: Optional[float] = 1e-8) -> None) -> None
            Nesterov Matrix Regressor
        
        Parameters
        ----------
        _lambda : float, optional
            Penalty parameter, default is `1.0`
        _delta : float, str, optional
            Provided Lipschitz constant `1 / L`, default is `'auto'`.\n
            When set to `'auto'`, it will use Fisher Imformation to estimate Lipschitz constant `L` 
            and set initial `delta` one or two orders of magnitude larger than `1 / L`
        loss : str, optional
            Loss function, default is `'neg-loglike'`\n
            `L(B) = \\frac{1}{2n}\\cdot \\sum_{i=1}^{n} (y_i - X_i\\cdot B)^2 + J(B)`
        penalty : str, optional
            Penalty method, default is `'nuclear'`
        max_steps : int, optional
            The max iteration steps for model optimization, default is `50`
        max_steps_epoch : int, optional
            The max iteration steps for each epoch during Armijo line search, default is `25`
        eps : float, optional
            Convergence threshold of objective function, default is `1e-8`
        """
        super().__init__(_lambda, _delta, penalty, loss, max_steps, max_steps_epoch, eps)

        # set task
        self.task = 'regression'
    
    # predict
    def predict(self, X : np.ndarray) -> np.ndarray:
        # dimension check
        assert(len(X.shape) == 3)
        assert(X.shape[1] == self.coef_.shape[0] and X.shape[2] == self.coef_.shape[1])

        return batch_mat_prod(X,self.coef_)

    # compute loss
    def loss(self, X : np.ndarray, y : np.ndarray, coef : np.ndarray) -> float:
        if self.loss_ == "neg-loglike":
            return 0.5 * np.mean((y - batch_mat_prod(X,coef))**2)

    # compute gradient of loss at point S
    def gradients(self, S : np.ndarray, X : np.ndarray, y : np.ndarray) -> np.ndarray:
        # batch size
        B = X.shape[0]
        if self.loss_ == "neg-loglike":
            grad = np.mean((y - batch_mat_prod(X,S)).reshape((B,1,1)) * (-X),axis=0)
        return grad

    def score(self, X : np.ndarray, y : np.ndarray, metric : Optional[str]=None) -> float:
        metric = "mse" if metric == None else metric
        y_pred = self.predict(X)
        if metric in ["mse","MSE","mean_square_error"]:
            s = mean_squared_error(y,y_pred)
        elif metric in ["me","max_error"]:
            s = max_error(y,y_pred)
        elif metric in ["mae","mean_absolute_error"]:
            s = mean_absolute_error(y,y_pred)
        elif metric == "loss":
            s = self.loss(X,y,self.coef_)
        else:
            raise ValueError("Not supported for metric: %s"%(metric))
        return s