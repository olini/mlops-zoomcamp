# 01 - Introduction

The ML project used will be a system to predict the duration of a taxi ride.

In a simple way, we can see 3 stages for a ML project:

1. Design: Is ML the right approach for the problem?
2. Train: Do some experiments and train some models to solve the problem
3. Operate: In this phase, we deploy the model, through an API, for example, to make it usable by users and business. Here, we also integrate and monitor the model.

MLOps is present on all the 3 stages. It is a set of practices to automate everything and enhance integration between a team working on a ML solution.

The topics we will cover which are important for a good ML development:

- Logging metrics and models during exploration and experimentation (Experiment Tracking and Model Registry)
- Deployment of the model, through a web service, for example
- Monitoring of the model, being able to detect when the performance is dropping and triggering a re-training process (this step can start manual and be improved to be fully automated when the system is trusted)
- Overall best practices for coding and code maintenance, like testing, documentation, modularization, etc

## ML Maturity Model

> Based on this [Microsoft’s article](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model) about the topic
> 

### 0. No MLOps

- No automation
- All code in Jupyter
- Usually in this case, the data scientists work alone, without any integration with the engineering team. It can be OK for a POC phase of the project.

### 1. DevOps, no MLOps

- Releases are automated
- Unit and integration tests, CI/CD
- Ops metrics
- Overall all the best practices for software engineering, without touching ML aspects
- Still no experiment tracking, no reproducibility. DS separated from engineers
- At this phase, you can go from POC to Production

### 2. Automated Training

- Training pipeline
- Experiment tracking
- Model registry
- Deployment still not necessarily automated, but already low friction
- DS already work with engineering
- The infrastructure required for this phase usually makes sense when the team already have multiple models in Production and need to scale. 1 or 2 models only in Prod you can still stay in level 1
- 2-3+ ML cases

### 3. Automated Deployment

- Easy to deploy model
- Usually you have a ML Platform, sends an API call to it and receives the URL where the model is deployed
- ML Platform should have features to support and host A/B Testing, serving to some users the model v1 and other model v2, logging the results
- Model monitoring

### 4. Full MLOps Automation

- No human interaction need on all the ML process deployment and retraining, it is all fully automated

In a company, not all models need to be in phase 3 or 4 automatically, first they need to be tested to see if the process is fine and trustful. Each project requires a certain level of automation.