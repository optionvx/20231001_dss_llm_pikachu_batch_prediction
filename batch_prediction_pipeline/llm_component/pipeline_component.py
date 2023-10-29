import os
import pandas as pd
import logging
from llm_label_flow import llm_component

class llm_pipeline_component:
    """A class for managing the NLP component of the LLM (Language Model)."""

    def __init__(self,
                 gcs_main_bucket: str,
                 gcs_folder: str,
                 project_id: str,
                 api_type: str,
                 labels: list,
                 llm_query: str,
                 file_suffix: str,
                 rate_limit: int = 20):
        """
        Args:
            gcs_main_bucket (str): The main Google Cloud Storage bucket.
            gcs_folder (str): The specific folder in the GCS bucket.
            project_id (str): The Google Cloud Project ID.
            api_type (str): Name of the api type to use.
            labels (list): List of labels to use.
            llm_query (str): Prompt for LLM
            file_suffix (str): Suffix to add to files.
            rate_limit (int): Rate limit for API calls/second

        """
        self.gcs_main_bucket = gcs_main_bucket
        self.gcs_folder = gcs_folder
        self.project_id = project_id
        self.api_type = api_type
        self.labels = labels
        self.llm_query = llm_query
        self.file_suffix = file_suffix
        self.rate_limit = rate_limit
        self.predictor = llm_component(query=self.llm_query, api_type=self.api_type, labels=self.labels, rate_limit=self.rate_limit)

    def llm_nlp_flow(self) -> None:
        """Execute the NLP workflow."""
        # Step 1: Generate labels for the data
        labelled_data = self.label_data()

        # Step 2: Construct the path where the JSON file will be saved
        json_file_path = f'llm_nlp_predictions_{self.file_suffix}.json'

        # Step 3: Save the labeled data to a JSON file and send to GCP
        labelled_data = labelled_data.drop_duplicates(subset='uid')
        labelled_data.to_json(json_file_path, lines=True, orient='records')
        os.system(f'gsutil -m cp {json_file_path} gs://{self.gcs_main_bucket}/{self.gcs_folder}/predictions/')

    def label_data(self) -> pd.DataFrame:
        """Generate labels for the data.

        Returns:
            pd.DataFrame: The labelled data.
        """
        # Step 1: Grab the data that needs labeling
        data = self.grab_data()

        # Step 2: Use the predictor to generate labels for the data
        results = self.predictor.generate_labels_langchain(data=data)
        results_df = pd.DataFrame.from_dict(results)
        return results_df

    def grab_data(self) -> pd.DataFrame:
        """Grab data from BigQuery filtered by the date range.

        Returns:
            pd.DataFrame: Filtered data.
        """
        # Create tmp
        os.makedirs('tmp')
        # Step 2: Grab data from GCS
        os.system(f'gsutil -m cp gs://{self.gcs_main_bucket}/{self.gcs_folder}/preprocessed/input_data.json tmp/')
        data = pd.read_json('tmp/input_data.json',orient='records',lines=True)
        df_dict = {'uid': list(data['uid']), 'text': list(data['text'])}
        logging.info('Grabbed data from GCS')
        return df_dict
