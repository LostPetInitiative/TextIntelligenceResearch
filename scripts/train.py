import json
import os
from pathlib import Path
from random import random, randint
import sys

from sklearn import metrics

from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from mlflow import log_metric, log_param, log_artifacts
from mlflow.sklearn import eval_and_log_metrics
import mlflow
import sklearn

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class InputAdapter(BaseEstimator, TransformerMixin):
    '''This one links MLFflow model (requires Pandas DF input for schema to be deployable model) with
        SKLearn text preprocessing (requires list of strings as input)
    '''
    def fit(self, x, y=None):
        return self

    def transform(self, x:pd.DataFrame):
        result = x["text"].to_list()
        #print(f"outputting {len(result)} entries. result is {type(result)}.")
        return result

messageTypesDesc = ["Lost","Found","NotRelevant/Other"]
speciesTypesDesc = ["Cat","Dog","Other"]
sexesDesc = ["Female","Male","NotDescribed/Other"]

def encodeDataSample(sample):
    messageTypes = ["L","F","NR"]
    speciesTypes = ["C","D","O"]
    sexes = ["F","M","O"]
    messageType = messageTypes.index(sample["MessageType"])
    species = speciesTypes.index(sample["Species"])
    sex = sexes.index(sample["Sex"])
    return (messageType, species, sex)
    
if __name__ == "__main__":
    dataFile = sys.argv[1]
    with open(dataFile,"r", encoding="utf-8") as inputFile:
        parsed = json.load(inputFile)
    print(f"Input data contain {len(parsed)} entries")

    taskNum = 0
    if taskNum == 0:
        taskName = "Lost/Found message"
        labelNames = messageTypesDesc
    elif taskNum == 1:
        taskName = "Described species"
        labelNames = speciesTypesDesc
    elif taskNum == 2:
        taskName = "Animal sex"
        labelNames = sexesDesc
    else:
        raise "Unsupported task num"

    
    

    texts = [x["text"] for x in parsed]
    targets = [labelNames[encodeDataSample(x)[taskNum]] for x in parsed]

    df = pd.DataFrame(texts, columns=["text"])

    X_train, X_test, y_train, y_test = train_test_split(df,targets, test_size=0.4, random_state=1433532, stratify=targets)

    count_vect  = CountVectorizer(input="content",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b"
    )

    sk_model = Pipeline([
        ('adapter', InputAdapter()),
        ('vect', count_vect),
        ('tfidf', TfidfTransformer()),
        #('mnb', MultinomialNB()),
        #('clf', ComplementNB()),
        ('linreg',OneVsRestClassifier(LinearRegression())),
        #('linSVC',OneVsRestClassifier(LinearSVC())),
        ## ("gauss",GaussianProcessClassifier())
        #("perceptron", Perceptron())
        #("randomforest",RandomForestClassifier())
    ])

    experiment = mlflow.set_experiment(
        f"{taskName} classifier of VK messages",
        #artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        #tags={"version": "v1", "priority": "P1"},
    )

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        description=f'Input {dataFile};\nmode {sk_model}') as run:
        
        #mlflow.sklearn.autolog()

        sk_model.fit(X_train, y_train)

        input_example = {
            "text": r"https://vk.com/id395768903 информационно. МО Сидит пёс метис в типе овчарки в ошейнике на платформе рн станции в Хотьково, напуган очень. Покормили, кушать не стала, не агрессивная, добрая.",
        }

        input_schema = Schema([
            ColSpec("string", "text"),
        ])
        output_schema = Schema([ColSpec("string", taskName)])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        mlflow.sklearn.log_model(sk_model, "sk_models", signature=signature, input_example = input_example)
        
        predicted = sk_model.predict(X_test)

        # metrics1 = eval_and_log_metrics(sk_model, X_test, y_test, prefix="val_")
        
        report = metrics.classification_report(y_test, predicted,  output_dict=True)

        print("Test dataset metrics:")
        print(report)
        # {'label 1': {'precision':0.5,
        #      'recall':1.0,
        #      'f1-score':0.67,
        #      'support':1},

        

        for i,(cl,cl_metrics) in enumerate(report.items()):
            if isinstance(cl_metrics,dict):
                for j,(metric_name,metric_value) in enumerate(cl_metrics.items()):
                    mlflow.log_metric(f"{cl}_{metric_name}",metric_value)
            elif isinstance(cl_metrics, float):
                mlflow.log_metric(cl,cl_metrics)
