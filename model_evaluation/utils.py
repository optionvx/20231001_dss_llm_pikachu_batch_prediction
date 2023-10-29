import pandas as pd
import numpy as np
from llm_label_flow import llm_component
from sklearn.metrics import classification_report

def run_llm_label_flow(label_dict, query, labels, data, true_label, label_name='llm_label'):
    """
        Runs the LLM label flow on a given dataset.

        Parameters:
        - label_dict (dict): A dictionary mapping the LLM labels to the true labels.
        - query (str): The query to run the LLM on.
        - labels (list): A list of labels to run the LLM on.
        - data (pd.DataFrame): The data to run the LLM on.
        - true_label (str): The column name of the true label.
        - label_name (str): The column name of the LLM label.

        Returns:
        - df: The original dataframe with the LLM labels appended.
        """

    df = data.copy()
    df_dict = {'uid': list(df['uid']), 'text': list(df['text'])}
    predictor = llm_component(query=query, api_type='vertex-api', labels=labels, rate_limit=20)
    results = predictor.generate_labels_langchain(data=df_dict)

    results_df = pd.DataFrame.from_dict(results)
    results_df['label'] = results_df['label'].apply(
        lambda x: label_dict[x] if x in label_dict.keys() else 'cannot_convert')
    results_df = results_df.loc[results_df['label'] != 'cannot_convert'].reset_index(drop=True)

    df = df.loc[df['uid'].isin(set(results_df['uid']))].reset_index(drop=True)
    result_dict = {uid: label for uid, label in zip(results_df['uid'], results_df['label'])}
    df[f'{label_name}'] = df['uid'].apply(lambda x: result_dict[x])
    df['true_label'] = df[f'{true_label}'].apply(lambda x: label_dict[x])

    print(classification_report(df['true_label'], df['llm_label']))

    return df


def disagreement_percentage(model1_preds, model2_preds, label_dict=False):
    """
    Calculate the percentage of disagreement between two model's predictions.

    Parameters:
    - model1_preds (list of str): Predictions from model 1.
    - model2_preds (list of str): Predictions from model 2.

    Returns:
    - dict: A dictionary containing the breakdown of percentage of disagreements.
    """
    if label_dict:
        model1_preds = [label_dict[x] for x in model1_preds]
        model2_preds = [label_dict[x] for x in model2_preds]

    # Ensure the predictions are of the same length
    if len(model1_preds) != len(model2_preds):
        raise ValueError("The predictions of both models should have the same length.")

    # Initialize a dictionary to store counts of disagreements
    disagreement_counts = {}

    # Loop through paired predictions and count disagreements
    for m1_pred, m2_pred in zip(model1_preds, model2_preds):
        if m1_pred != m2_pred:
            key = (m1_pred, m2_pred)
            disagreement_counts[key] = disagreement_counts.get(key, 0) + 1

    # Convert counts to percentages
    total_disagreements = sum(disagreement_counts.values())
    print(f'The number of disagreements were: {total_disagreements}')
    print(
        f'Out of the {len(model1_preds)} samples the models disagreed: {np.round((total_disagreements / len(model1_preds)) * 100, 2)}% of the time')
    for key, count in disagreement_counts.items():
        disagreement_counts[key] = (count / total_disagreements) * 100

    return disagreement_counts
