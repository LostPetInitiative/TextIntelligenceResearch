import json
import os
from pathlib import Path
from random import random, randint
import sys

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from mlflow import log_metric, log_param, log_artifacts
from mlflow.sklearn import eval_and_log_metrics
import mlflow

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
        taskName = "Lost/Found message classifier"
        labelNames = messageTypesDesc
    elif taskNum == 1:
        taskName = "Described species classifier"
        labelNames = speciesTypesDesc
    elif taskNum == 2:
        taskName = "Animal sex classifier"
        labelNames = sexesDesc
    else:
        raise "Unsupported task num"

    texts = [x["text"] for x in parsed]
    targets = [encodeDataSample(x)[taskNum] for x in parsed]

    X_train, X_test, y_train, y_test = train_test_split(texts, targets, test_size=0.4, random_state=1433532, stratify=targets)

    count_vect  = CountVectorizer(input=X_train,
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b"
    )

    sk_model = Pipeline([
        ('vect', count_vect),
        ('tfidf', TfidfTransformer()),
        ('clf', ComplementNB()),
        #('linreg',LinearRegression()),
    ])

    experiment = mlflow.set_experiment(
        f"{taskName} of VK messages",
        #artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        #tags={"version": "v1", "priority": "P1"},
    )

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        
        #mlflow.sklearn.autolog()

        p = sk_model.fit(X_train, y_train)

        mlflow.sklearn.log_model(sk_model, "sk_models")

        predicted = sk_model.predict(X_test)

        # metrics1 = eval_and_log_metrics(sk_model, X_test, y_test, prefix="val_")
        
        report = metrics.classification_report(y_test, predicted, target_names=labelNames, output_dict=True)

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
