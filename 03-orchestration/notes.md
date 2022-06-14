# 03 - Orchestration and ML Pipelines

# Workflow Orchestration

Basically a set of tools to organize and monitor a workflow of interest.

In a ML pipeline, there are many points that may fail, that is why we need to monitor and try to minimize these problems. That is why we need workflow orchestration, so we can handle errors in a more transparent and organized way.

## Negative Engineering

Coding to handle errors and do workarounds. 90% of engineering time spent on retries when APIs go down, malformed data, notifications, timeouts, etc. That is where Prefect comes to reduce this time spent so Data Scientists can focus more time into business.

# Introduction to Prefect

> Eliminating Negative Engineering
> 
- Dynamic DAG-free workflows
- Transparent and observable orchestration rules

> Quick note on Parquet: it is around 1/5 the size of csv, so it has some compression. But also it holds schema! A very convenient feature that csv doesn’t have and can save a lot of time converting and assuring data is on the right format.
>