#20231001_dss_llm_pikachu_batch_prediction

# What
This repo contains the code for the batch prediction pipeline for the DSS Meetup. The pipeline is built using Kubeflow Pipelines and Vertex AI.

# Getting Started

This repo is broken down into two parts. You will want to start in order:
1. Model Evaluation Directory (model_evaluation/)
>> This directory contains the code for model evaluation. Please refer to model_evaluation_guide.md for more instructions.

2. Batch Prediction Pipeline Directory (batch_prediction_pipeline/)
>> This directory contains the code for batch prediction pipeline. Please refer to batch_prediction_pipeline_guide.md for more instructions.


# Other Things to Note Before Running the Code

1. Virtual Envs: You're going to want to create two separate virtual environments to run the evaluation code and pipeline build code.
Detailed instructions are listed in each markdown file in the respective directories.

2. Ensure that you have all the necessary projects set up in your Google Cloud Platform Project. For more details follow the instructions here: https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines



