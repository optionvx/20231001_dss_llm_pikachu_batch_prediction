import traceback
import logging
from utils import parse_arguments
from pipeline_component import llm_pipeline_component


if __name__ == '__main__':

    try:
        args = parse_arguments()
        custom_models_component = llm_pipeline_component(args.gcs_main_bucket, args.gcs_folder, args.project_id,args.api_type,args.labels, args.llm_query, args.file_suffix, args.rate_limit).llm_nlp_flow()
        logging.info('Prediction complete')
    except Exception as e:
        # print the full traceback
        traceback.print_exc()
        # raise a new exception with a custom message'
        raise Exception("An error occurred: " + str(e))

