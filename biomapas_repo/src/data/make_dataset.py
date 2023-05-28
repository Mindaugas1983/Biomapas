# -*- coding: utf-8 -*-
import logging
import click
import json
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def filtering_data(file_path):
    """
        This function reads given raw data file and extracts user questions, answers, and contexts and returns this data
        Args:
           file_path': path where raw json data file is located
        Returns:
            filtered_data: list of dictionaries with filtered data
    """
    filtered_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data['questions']:
            question = item['body']
            contexts = [context['text'] for context in item['snippets']]
            answer = item['ideal_answer']
            filtered_data.append({
                'question': question,
                'answer': answer,
                'contexts': contexts
            })
        return filtered_data


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
        This function gets two arguments from command line, loads given data json file
        filters data and leaves only actual data. This data ias saved as a file that ready to be analyzed (in ../processed).
        Args:
           input_filepath': command line argument which represents path whereraw data file is stored
           output_filepath:  command line argument which represents path where results should be stored
        Raises:
            If there is no two command line arguments when running this file or if there are no file with raw data exception is raised
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    medical_data = filtering_data(input_filepath)
    with open(output_filepath, 'w') as f:
        json.dump(medical_data, f)


if __name__ == '__main__':
    """
      On start execution of this python file we are turning on logging loading enviromental variables if there 
      are some and running main() function 
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()

