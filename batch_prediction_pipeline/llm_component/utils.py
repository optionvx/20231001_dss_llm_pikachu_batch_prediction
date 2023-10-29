import argparse
import datetime

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs-main-bucket',
                        type=str,
                        default='your-main-bucket',
                        help='Main GCS Bucket w/ og data.')
    parser.add_argument("--gcs-folder", type=str, default="your-gcs-folder")
    # CHANGE COMPONENT 1 TO GCS-MAIN AND GCS-NLP
    parser.add_argument("--project-id", type=str, default='your-project-id')
    parser.add_argument("--api-type", type=str, default='vertex-api')
    parser.add_argument("--labels", type=list, default=['positive','negative','neutral'])
    parser.add_argument("--llm_query", type=str, default='Label the data as positive, negative, or neutral')
    parser.add_argument("--file-suffix", type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    parser.add_argument("--rate-limit", type=int, default=20)
    args = parser.parse_known_args()[0]
    return args

