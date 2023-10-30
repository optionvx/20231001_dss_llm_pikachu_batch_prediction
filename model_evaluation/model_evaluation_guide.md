# Model Evaluation Guide 


### Step 1. Create Virtual Environment
To be able to recreate local runs of this stuff, it may come in handy to recreate the environment.

```
cd /model_evaluation 
conda create -n pikachu_model_eval_env python=3.10
conda activate pikachu_model_eval_env
pip3 install -r requirements.txt 
python -m ipykernel install --user --name=pikachu_model_eval_env # add to your jupyter kernel for running code in model_evaluation_notebook.ipynb

```
### Step 2. Follow along in the notebook
You can follow along in the notebook `model_evaluation_notebook.ipynb`. This notebook covers two things:
1. Prompt Engineering
2. Final Model Assessment 

### Other Things of Note 

The class you will be working with throughout this whole tutorial is llm_component defined in llm_label_flow.py in this directory. 
Please refer to the arguments of the class for your specific use case. 

**Rate Limit**:
>> If your plan is defined as x requests/minute then the rate limit is 60/x requests/second. E.g. 1200 requests/min * 1 min/60 seconds = 20 requests/second.

**API Types**: 
>> Vertex AI API: api_type = 'vertex-api'

>> PaLM API: api_type = 'palm-api'

>> OpenAI API: api_type = 'openai-api'

**Model Parameters**:
>> Defaults are used in the class with the exception of a temperature set to 0 (to force the LLM to become more deterministic). Please adjust to fit your specific use case.

