# Batch Prediction Pipeline Guide  

<img src="img/pika_pic.png" alt="drawing" width="80"/> <img src="img/vertexai.png" alt="drawing" width="50"/>
<img src="img/kubeflow.png" alt="drawing" width="50"/>
<img src="img/langchain.png" alt="drawing" width="100"/>

Please ensure all necessary projects are set up in your Google Cloud Platform Project. For more details follow the instructions here: https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines 
### Step 1. Create Virtual Environment

For Kubeflow Pipeline Build in pipeline_build.ipynb
```
cd /batch_prediction_pipeline 
conda create -n pikachu_pipeline_build python=3.7
conda activate pikachu_pipeline_build
pip3 install -r requirements.txt 
python -m ipykernel install --user --name=pikachu_pipeline_build # add to your jupyter kernel for pipeline_build.ipynb
```


### Step 2. Build out the container image and send to GCS Container Registry for Kubeflow Pipeline

Note this is just a base docker image for a tutorial. Security guidelines should be set to best fit your use case within the docker container.

1. 
>> Ensure that you have docker installed. In the llm_component directory run the following command in the terminal inside the directory to build the docker image. 
```
image=gcr.io/<your-project-id>/<your-pipeline-name>/llm_component
docker build -t $image .
```

2. 
>> Send out the docker image to the Google Cloud Platform Container Registry. 
```
docker push $image
```

### Step 3. Build the Kubeflow Pipeline in batch_prediction_pipeline/pipeline_build.ipynb
>> Follow the instructions in the notebook to build the pipeline. Ensure you replace the pipeline parameter values with your own. 






