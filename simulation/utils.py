# function used to run simulations
import numpy as np
import pandas as pd
import sys
from typing import Union, Callable, Optional
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append("D:\Code\MatrixGLM")
from model.MatrixTransfer import TransMatrixGLM
from model.utils import multi_task

# create model instance
def CreateModel(model : str, task : str="regression", max_steps : int=100,**kwargs):
    if model == "naive-Lasso":
        return TransMatrixGLM(task=task,penalty_debias="lasso",max_steps=max_steps,**kwargs)
    elif model == "naive-Nuclear":
        return TransMatrixGLM(task=task,penalty_debias="nuclear",max_steps=max_steps,**kwargs)
    elif "Lasso-Lasso" in model:
        return TransMatrixGLM(task=task,penalty_transfer="lasso",penalty_debias="lasso",max_steps=max_steps,**kwargs)
    elif "Nuclear-Lasso" in model:
        return TransMatrixGLM(task=task,penalty_transfer="nuclear",penalty_debias="lasso",max_steps=max_steps,**kwargs)
    elif "Nuclear-Nuclear" in model:
        return TransMatrixGLM(task=task,penalty_transfer="nuclear",penalty_debias="nuclear",max_steps=max_steps,**kwargs)

# compute Frobenius-norm metric
def metrics(
    model : str, task : str, A_set : list, lambda_transfer : float, lambda_debias : float,
    X_target : np.ndarray, y_target : np.ndarray, X_source : list, y_source : list, B : np.ndarray):
    """
    This function is used to assist multi process acceleration
    """
    estimator = CreateModel(
        model=model,task=task,lambda_transfer=lambda_transfer,lambda_debias=lambda_debias)
    tic = time.time()
    estimator.fit(Xt=X_target,yt=y_target,Xa=X_source,ya=y_source,A=A_set)
    toc = time.time()
    err = np.linalg.norm(B - estimator.coef_,ord="fro")
    score = estimator.score(X_target,y_target)
    time_used = toc - tic
    return err, score, time_used

# compute Frobenius-norm metric
def metrics_helper(
    model, task, A_set, lambda_transfer, lambda_debias,
    X_target, y_target, X_source, y_source, B, h, K, repeat):
    """
    This is the decoration of the `metrics` function,\n 
    so that the return value contains the configuration information of the input parameter.
    """
    res = {
        "model": model,
        "h": h,
        "K": K,
        "repeat": repeat,
        "lambda_transfer": lambda_transfer,
        "lambda_debias": lambda_debias,
        "metrics": metrics(model,task,A_set,lambda_transfer,lambda_debias,X_target,y_target,X_source,y_source,B)
    }
    return res

def Sec41_simulation(
    models : list, coef_func : Callable, data_func : Callable, task : str,
    h_list : list, Kmax : int=10, replicate : int=100, tuning_rounds : int=3,
    lambda_min_ratio : float=0.01, lambda_max_ratio : float=5, 
    metric : str=None, scale : str="none",n_jobs : int=16, top : int=1, optimal_lambda : Optional[dict]=None, base : Optional[bool]=True):
    # init results
    result = pd.DataFrame(data=None,columns=["h","K","repeat","method","err","score"])
    penaltoes = pd.DataFrame(data=None,columns=["h","K","method","transfer","debias"])
    # show progress
    pbar = tqdm(total=len(h_list)*Kmax*len(models),ncols=120)
    tuning = True if optimal_lambda is None else False
    for h in h_list:
        for K in range(1,Kmax+1):
            # find optimal penalty coef
            if tuning:
                optimal_lambda = {}
                for m in models:
                    optimal_lambda["transfer_%s"%(m)] = 0 
                    optimal_lambda["debias_%s"%(m)] = 0
                for m in models:
                    pbar.set_description("Tuning on h: %4d, K: %2d, model: %s"%(h,K,m))
                    A_set = None if "naive" in m else list(range(K))
                    for i in range(tuning_rounds):
                        B, W = coef_func(h=h,K=K)
                        X_target, X_source, y_target, y_source = data_func(B=B,W=W,scale=scale)
                        model = CreateModel(model=m,task=task)
                        best_lambda_transfer, best_lambda_debias, _, _ =\
                            model.tuning(
                                Xt=X_target,yt=y_target,Xa=X_source,ya=y_source,A=A_set,
                                lambda_min_ratio=lambda_min_ratio,lambda_max_ratio=lambda_max_ratio,
                                metric=metric,top=top,n_jobs=n_jobs)
                        optimal_lambda["transfer_%s"%(m)] += best_lambda_transfer / tuning_rounds
                        optimal_lambda["debias_%s"%(m)] += best_lambda_debias / tuning_rounds
                    penaltoes.loc[len(penaltoes),:] =\
                     [h,K,m,optimal_lambda["transfer_%s"%(m)],optimal_lambda["debias_%s"%(m)]]
            # replications
            for m in models:
                # update progress
                pbar.set_description("Fitting on h: %4d, K: %2d, model: %s"%(h,K,m))
        
                # create a parameter list for parallel execution
                A_set = None if "naive" in m else list(range(K))
                param_list = []
                for repeat in range(replicate):
                    B, W = coef_func(h=h,K=K)
                    X_target, X_source, y_target, y_source = data_func(B=B,W=W,scale=scale)
                    lambda_transfer = optimal_lambda["transfer_%s"%(m)] * np.sqrt(np.log(64*64)/(len(X_target) + K*len(X_source[0]))) if base else optimal_lambda["transfer_%s"%(m)]
                    lambda_debias = optimal_lambda["debias_%s"%(m)] * np.sqrt(np.log(64*64)/(len(X_target) )) if base else optimal_lambda["debias_%s"%(m)]
                    param = {
                        "model": m, "task": task, "A_set": A_set, 
                        "lambda_transfer": lambda_transfer,
                        "lambda_debias": lambda_debias,
                        "X_target": X_target, "y_target": y_target,
                        "X_source": X_source, "y_source": y_source,
                        "B": B, "h": h, "K": K, "repeat": repeat+1
                    }
                    param_list.append(param.copy())
                # multi-process accelerated
                res = multi_task(metrics_helper,param_list,n_job=n_jobs,verbose=False)
                # save result
                for i,r in enumerate(res):
                    row = len(result)
                    result.loc[row,"h"] = r["h"]
                    result.loc[row,"K"] = r["K"]
                    result.loc[row,"repeat"] = r["repeat"]
                    result.loc[row,"method"] = r["model"]
                    result.loc[row,"err"] = r["metrics"][0]
                    result.loc[row,"score"] = r["metrics"][1]
                    result.loc[row,"time"] = r["metrics"][2]
                pbar.update(1)

    pbar.clear()
    pbar.close()
    return result, penaltoes

def Sec41_simulation_cv(
    models : list, coef_func : Callable, data_func : Callable, task : str,
    h_list : list, Kmax : int=10, replicate : int=100,
    lambda_min_ratio : float=0.01, lambda_max_ratio : float=5, 
    metric : str=None, scale : str="none",n_jobs : int=16, folds : int=4, top : int=1):
    # init results
    result = pd.DataFrame(data=None,columns=["h","K","repeat","method","err","score","lambda_transfer","lambda_debias"])
    # show progress
    pbar = tqdm(total=len(h_list)*Kmax*len(models)*replicate,ncols=120)
    for h in h_list:
        for K in range(1,Kmax+1):
            for repeat in range(replicate):
                B, W = coef_func(h=h,K=K)
                X_target, X_source, y_target, y_source = data_func(B=B,W=W,scale=scale)
                for m in models:
                    pbar.set_description("Tuning on h: %4d, K: %2d, round: %4d, model: %s"%(h,K,repeat+1,m))
                    A_set = None if "naive" in m else list(range(K))
                    model = CreateModel(model=m,task=task)
                    best_lambda_transfer, best_lambda_debias, _, _ =\
                    model.tuning(
                        Xt=X_target,yt=y_target,Xa=X_source,ya=y_source,A=A_set,
                        lambda_min_ratio=lambda_min_ratio,lambda_max_ratio=lambda_max_ratio,
                        metric=metric,n_jobs=n_jobs,folds=folds,top=top)
                    err = np.linalg.norm(B - model.coef_,ord="fro")
                    score = model.score(X_target,y_target)
                    # add results
                    row = len(result)
                    result.loc[row,:] = [h,K,repeat+1,m,err,score,best_lambda_transfer,best_lambda_debias]
                    
                    pbar.update(1)
    pbar.clear()
    pbar.close()
    return result

# plot simulation results
def plot_simulation(result,h_list):
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ax = ax.flatten()
    sns.set_style("darkgrid")
    sns.set_context("paper")
    for i, h in enumerate(h_list):
        data = pd.melt(frame=result.loc[result["h"] == h],id_vars="K",value_vars=result.columns[3:])
        data.columns = ["K","method","Frobenius-norm"]
        l = sns.pointplot(data=data,x="K",y="Frobenius-norm",hue="method",ax=ax[i],palette="bright",ci=99,errwidth=1.5)
        ax[i].set_title("Linear Case ($h$=%d)"%(h))
        ax[i].get_legend().remove()
        ax[i].set_xticks(list(range(0,10)))
        ax[i].set_xticklabels(list(range(1,11)))
    lines=[]
    labels=[]
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    fig.legend(
        lines[0:5], labels[0:5],loc="lower center",
        bbox_to_anchor=(0.5, -0.05), ncol=5, 
        title=None, frameon=False,edgecolor="red")
    plt.tight_layout()