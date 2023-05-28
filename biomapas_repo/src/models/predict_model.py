import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import pickle
from train_model import PredictionModel


def make_contexts(u_questions):
    """
       This function takes a list of user input and prepares list of contexts: this user question and all
       previous user questions are treated as a context
       Args:
         u_questions: list of user inputs stored as strings
       Returns:
         u_contexts: list of user question context for each user question
    """

    u_contexts = []
    for idx, _ in enumerate(u_questions):
        u_contexts.append('. '.join(u_questions[:idx+1]))
    return u_contexts


@click.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.argument('user_input_file', type=click.Path(exists=True))
def main(model_file, user_input_file):
    """
     This function gets two arguments from command line, loads user question file stored as text, loads
     saved prediction FAQ prediction model prepares list of user contexts and calls model.predict_faq()
     function with prepared data
     Args:
       model_file: command line argument which represents path where serialized model file is stored
       user_input_file:  command line argument which represents path where text file with user questions is stored
     Raises:
       If there is no two command line arguments when running this file or if there are no files in  given paths exceptions are raised
       If data from user input file cant be loaded exception is raised: "User questions was not received"
       If prediction model cant be loaded exception is raised: 'Prediction model has been found and loaded'
     """
    try:
        with open(user_input_file) as f:
            u_questions = []
            for line in f:
                u_questions.append(line)
    except("User questions was not received"):
        return
    try:
        with open(model_file + '', 'rb') as f:
            loaded_model = pickle.load(f)
            print('Prediction model has been found and loaded')
    except('Error occured when loading model'):
        return
    u_contexts = make_contexts(u_questions)
    loaded_model.predict_faq(u_questions, u_contexts, num_suggestions=3)


if __name__ == '__main__':
    """
      On start execution of this python file we are turning on logging loading enviromental variables if there 
      are some and running main() function 
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()