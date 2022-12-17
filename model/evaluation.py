import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from moabb.paradigms import BaseParadigm, P300, SinglePass
from moabb.datasets.base import BaseDataset
from typing import Optional, Union, Callable, List, Dict, Any
from tqdm import tqdm
from abc import abstractmethod
import os
import time

# log = logging.getLogger(__name__)

class Graph(BaseEstimator):
    @abstractmethod
    def __init__(self, name : str, transfer : str, *args, **kwargs) -> None:
        self.name = name
        self.transfer_type = transfer

    @abstractmethod
    def fit(self, X_target : np.ndarray, y_target : np.ndarray,
            X_source : Optional[np.ndarray]=[], y_source : Optional[np.ndarray]=[],
            n_jobs : Optional[int]=1, random_state : Optional[int]=None, *args, **kwargs):
        """
        Parameters
        ----------
        X_target : np.ndarray
            The observations in target domain.
        y_target : np.ndarray
            The labels in target domamin.
        X_source : np.ndarray
            The observations in source domain.
        y_source : np.ndarray
            The labels in source domain.
        n_jobs : int, default = `1`
            Number of jobs for fitting of model.
        random_state : int, default = `None`
            Specify possible randomization seeds in the model.
        """
        ...
    
    @abstractmethod
    def predict(self, X : np.ndarray):
        ...
    
    @abstractmethod
    def predict_proba(self, X : np.ndarray):
        ...

def CrossSubjectEvaluation(
    X : np.ndarray,
    y : np.ndarray,
    metadata : pd.Series,
    models : List[Graph],
    suffix : str,
    subjects : List[int],
    eval_subjects : List[int]=None,
    save_path : Optional[str]='./',
    random_state : Optional[int]=None,
    n_jobs : Optional[int]=1) -> pd.DataFrame:
    """
    Cross Subject Evaluation
    ----------
    Evaluate model performance on each subject group.\n
    Model will train on the other subjects and evaluate on the left-one subject.\n

    Parameters
    ----------
    X : np.ndarray
        The observations.
    y : np.ndarray
        The labels.
    metadata : pd.Series
        The metadata used to identify and locate each sample.
    models : list of model
        The list of model used.
    suffix : str
        Suffix for the results file.
    subjects : List of in
        The subjects' id used for training.
    eval_subjects : List of int, default = `None`
        The subjects' id to be evaluated, i.e. the `target` subjects we care\n
        If `None`, will evaluate on all subjects
    save_path : str, default is `'./'`
        Specific path for storing the results.
    random_state : int, default = `None`
        If not `None`, can guarantee same seed for shuffling examples.
    n_jobs : int, default = `1`
        Number of jobs for fitting of pipeline.
    """
    # init results
    results = pd.DataFrame(data=None,columns=["model","transfer","subject","session","samples","accuracy","roc_auc","time"])

    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    eval_subjects = subjects if eval_subjects is None else eval_subjects
    n_subjects = len(eval_subjects)

    # perform Leave one subject out CV
    pbar = tqdm(total=n_subjects,ncols=100) # show progress
    
    # iterate over subjects
    for subject in eval_subjects:
        # update progress
        pbar.set_description("Processing Subject: %d"%(subject))

        # fetch test index
        test = groups == subject
        train_subjects, target_subjects = subjects.copy(), eval_subjects.copy()
        # remove test subject
        train_subjects.remove(subject) 
        target_subjects.remove(subject)
        # create training data
        X_target, X_source, y_target, y_source = get_source_target(X,y,train_subjects,groups,target_subjects)

        # iterate over models
        for model in models:
            pbar.set_description("Processing Subject: %d, training model: %s"%(subject,model.name))
            tic = time.time()
            model.fit(X_target,y_target,X_source,y_source,n_jobs=n_jobs,random_state=random_state)
            toc = time.time()

            # eval on each session
            for session in np.unique(sessions[test]):
                pbar.set_description("Processing Subject: %d, evaluate on: %s"%(subject,session))
                sess_id = (sessions == session) & test
                X_test, y_test = X[sess_id], y[sess_id]

                # evaluation
                acc = accuracy_score(y_true=y_test,y_pred=model.predict(X_test))
                auc = roc_auc_score(y_true=y_test,y_score=model.predict_proba(X_test))

                res = {
                    "model": model.name,
                    "transfer": model.transfer_type,
                    "subject": subject,
                    "session": session,
                    "samples": len(X_source) + len(X_target),
                    "accuracy": acc,
                    "roc_auc": auc,
                    "time": toc - tic,
                }
                # add to results
                results.loc[len(results),:] = res
        # update
        pbar.update(1)
    pbar.clear()
    pbar.close()
    
    # save results
    suffix = "" if suffix is None else suffix
    results.to_csv(os.path.join(save_path,suffix + "_" \
    + time.strftime("%Y-%m-%d %H-%M", time.localtime(time.time())) + ".csv"),index=False)

    return results

def CrossSessionEvaluation(
    X : np.ndarray,
    y : np.ndarray,
    metadata : pd.Series,
    models : List[Graph],
    suffix : str,
    subjects : List[int],
    eval_subjects : List[int]=None,
    save_path : Optional[str]='./',
    random_state : Optional[int]=None,
    n_jobs : Optional[int]=1) -> pd.DataFrame:
    """
    Cross Session Evaluation
    ----------
    Evaluate model performance on each session group.\n
    Model will train on the other sessions and evaluate on the left-one session.\n

    Parameters
    ----------
    X : np.ndarray
        The observations.
    y : np.ndarray
        The labels.
    metadata : pd.Series
        The metadata used to identify and locate each sample.
    models : list of model
        The list of model used.
    suffix : str
        Suffix for the results file.
    subjects : List of in
        The subjects' id used for training.
    eval_subjects : List of int, default = `None`
        The subjects' id to be evaluated, i.e. the `target` subjects we care\n
        If `None`, will evaluate on all subjects
    save_path : str, default is `'./'`
        Specific path for storing the results.
    random_state : int, default = `None`
        If not `None`, can guarantee same seed for shuffling examples.
    n_jobs : int, default = `1`
        Number of jobs for fitting of pipeline.
    """
    # init results
    results = pd.DataFrame(data=None,columns=["model","transfer","subject","session","samples","accuracy","roc_auc","time"])

    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    eval_subjects = subjects if eval_subjects is None else eval_subjects
    n_subjects = len(eval_subjects)

    # perform Leave one subject out CV
    pbar = tqdm(total=n_subjects,ncols=100) # show progress
    
    # iterate over subjects
    # iterate over sessions
    for sess in np.unique(sessions):
        # update progress
        pbar.set_description("Processing Session: %s"%(sess))

        # fetch train and test index
        test = sessions == sess
        train_subjects, target_subjects = subjects.copy(), eval_subjects.copy()
        # create training data
        X_target, X_source, y_target, y_source = \
            get_source_target(X[~test],y[~test],train_subjects,groups[~test],target_subjects)
        
        # iterate over models
        for model in models:
            pbar.set_description("Processing Session: %s, training model: %s"%(sess,model.name))
            tic = time.time()
            model.fit(X_target,y_target,X_source,y_source,n_jobs=n_jobs,random_state=random_state)
            toc = time.time()

            # eval on each subject
            for subject in eval_subjects:
                pbar.set_description("Processing Session: %s, evaluate on: %s"%(sess,str(subject)))
                subject_id = (groups == subject) & test
                X_test, y_test = X[subject_id], y[subject_id]

                # evaluation
                acc = accuracy_score(y_true=y_test,y_pred=model.predict(X_test))
                auc = roc_auc_score(y_true=y_test,y_score=model.predict_proba(X_test))

                res = {
                    "model": model.name,
                    "transfer": model.transfer_type,
                    "subject": subject,
                    "session": sess,
                    "samples": len(X_source) + len(X_target),
                    "accuracy": acc,
                    "roc_auc": auc,
                    "time": toc - tic,
                }
                # add to results
                results.loc[len(results),:] = res

        # update
        pbar.update(1)
    pbar.clear()
    pbar.close()
    
    # save results
    suffix = "" if suffix is None else suffix
    results.to_csv(os.path.join(save_path,suffix + "_" \
    + time.strftime("%Y-%m-%d %H-%M", time.localtime(time.time())) + ".csv"),index=False)

    return results


# get source data and target data
def get_source_target(X : np.ndarray, y : np.ndarray, subjects : list, 
                      groups : pd.Series, target_subjects) -> tuple:
    """
    sperate the data into source domain and target domain.

    Parameters
    ----------
    X : np.ndarray
        The observations
    y : np.ndarray
        The labels
    subjects : list
        The subjects used for training
    groups : pd.Series
        Series used to determine which group each sample belongs to
    target_subjects : list
        The subjects in target domain
    """
    # init
    X_source, X_target, y_source, y_target = [], [], [], []

    # construct training data
    for sub_id in subjects:
        if sub_id in target_subjects:
            X_target.append(X[groups == sub_id])
            y_target.append(y[groups == sub_id])
        else:
            X_source.append(X[groups == sub_id])
            y_source.append(y[groups == sub_id])
    X_target, y_target = np.concatenate(X_target), np.concatenate(y_target)
    if len(X_source) > 0:
        X_source, y_source = np.concatenate(X_source), np.concatenate(y_source)
    
    return X_target, X_source, y_target, y_source