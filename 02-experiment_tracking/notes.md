# 02 - Experiment Tracking

## Important concepts

- ML experiment: the process of bulding an ML model
- Experiment run: each trial in an ML experiment
- Run artifact: any file that is associated with an ML run
- Experiment metadata

# What is Experiment Tracking?

Experiment tracking is the process of keeping track of all the relevant information from an ML experiment, like: source code, env, data, model, hyperparams, metrics, etc

# Why is it so important?

- Reproducibility
- Organization
- Optimization

The more basic experiment tracking system that there is are spreadsheets. They are not enough because they are error prone, they have no standard format, and they are bad for visibility and collaboration. They bring not enough information about the experiments.

# A great tool for it: MLflow

> An open source platform for the machine learning lifecycle. [Official docs](https://mlflow.org)
> 

It contains 4 main modules:

- Tracking
- Models
- Model Registry
- Projects

## Tracking experiments with MLflow

The MLflow Tracking module allows you to organize your experiments into runs, and to keep track of:

- Parameters (everything used to train the model, like preprocess, path to files, etc)
- Metrics
- Metadata (Tags, like name of developer, type of algorithm)
- Artifacts (Graphs for visualization of results)
- Models

Along this information, MLflow automatically logs extra information about the run:

- Source code
- Version of the code (git commit)
- Start and end time
- Author

→ Command to run MLflow with backend storage using sqlite:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Model management

You can do model management manually, in folders for example. The problem with this is that it is error prone, it has no versioning and no model lineage. That is why MLflow for model management is a great choice.

## Model registry

In a scenario that you have a new model version and needs to update it in production, or even you need to retrain or rollback the model version. This could be done sending the model via email to the ML Engineer/MLOps Engineer. But this approach could lead to many errors and be very time consuming, that is why we need to use a Model Registry tool, as MLflow is.

The model registry module doesn’t deploy any models, it just registers which models are production ready to a better control and management. One would need to implement a CI/CD phase after model registry to deploy the model.