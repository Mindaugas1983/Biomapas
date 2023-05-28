
import logging
import json
import spacy
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import numpy as np
from transformers import AlbertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import copy
import pickle
from datetime import date


def load_data_from_json(file_path):
    """
       This function loads filtered JSON file and prepares questions answers and contexts lists
       Args:
         file_path: path to filtered JSON file, where questions answers and contexts are located
       Returns:
         questions: list of FAQ questions
         answers: list of FAQ answers
         contexts: list of FAQ contexts
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = []
    answers = []
    contexts = []
    for item in data:
        questions.append(item['question'])
        answers.append(str(item['answer']))
        contexts.append(str(item['contexts']))
    return questions, answers, contexts


def preprocess_data(qst, ctx, tokenizer, lemmatizator):
    """
       This function filters questions and contexts list from special symbols, lemmatizes and tokenizes them
       and extracts attention masks for these tokens, converts everything to numpy arrays and returns them
       Args:
         qst: list of FAQ questions
         ctx: list of FAQ contexts
         tokenizer: initiated tokenizer for tokenizing data
         lemmatizator: initiated lemmatizer for lemmatizing data

       Returns:
         Tokenized numpy array of FAQ questions
         numpy array of attention mask of tokenized FAQ questions
         Tokenized numpy array of FAQ contexts
         numpy array of attention mask of tokenized FAQ contexts
    """
    qst2 = copy.deepcopy(qst)
    ctx2 = copy.deepcopy(ctx)
    qst_filtered = [re.sub('[^a-zA-Z0-9]+', '', _.lower()) for _ in qst2]
    ctx_filtered = [re.sub('[^a-zA-Z0-9]+', '', _.lower()) for _ in ctx2]
    qst_lemmatized = []
    for item in qst_filtered:
        stc = lemmatizator(item)
        stc_lemmatized = " ".join([token.lemma_ for token in stc])
        qst_lemmatized.append(stc_lemmatized)
    ctx_lemmatized = []
    for item in ctx_filtered:
        stc = lemmatizator(item)
        stc_lemmatized = " ".join([token.lemma_ for token in stc])
        ctx_lemmatized.append(stc_lemmatized)
    qst_info = tokenizer.batch_encode_plus(qst_lemmatized, truncation=True, padding=True, return_attention_mask=True)
    ctx_info = tokenizer.batch_encode_plus(ctx_lemmatized, truncation=True, padding=True, return_attention_mask=True)
    qst_ids = qst_info['input_ids']
    qst_mask = qst_info['attention_mask']
    ctx_ids = ctx_info["input_ids"]
    ctx_mask = ctx_info['attention_mask']
    return np.array(qst_ids, dtype=int), np.array(qst_mask, dtype=int), np.array(ctx_ids, dtype=int), np.array(ctx_mask, dtype=int)


class PredictionModel():
    """
    Class with all required information from FAQ saved as self variables, with method which evaluates given user
    questions and contexts inputs  and assigns most suitable questions and answers from FAQ

    Methods:
        _preprocess_user_data: inner method which prepares user input for evaluation
        predict_faq: method which prepares given user data and assigns requested number of most suitable FAQ questions
                            and answers for each user question
    """
    def __init__(self, qst_ids, qst_mask, ctx_ids, ctx_mask, qst, ans, ctx, qst_length, ctx_length):
        """
        Class initialization method which stores all required information for user input evaluation
        Args:
          qst_ids: numpy array of tokenized FAQ questions
          qst_mask: numpy array of attention mask of tokenized FAQ questions
          ctx_ids: numpy array of tokenized FAQ contexts
          ctx_mask: numpy array of attention mask of tokenized FAQ contexts
          qst: list of FAQ questions
          ans: list of FAQ answers
          ctx: list of FAQ contexts
          qst_length: variable that shows with how much tokens user questions data needs to be tokenized
          ctx_length: variable that shows with how much tokens user contexts data needs to be tokenized
        """
        self.qst_ids = qst_ids
        self.qst_mask = qst_mask
        self.ctx_ids = ctx_ids
        self.ctx_mask = ctx_mask
        self.qst = qst
        self.ans = ans
        self.ctx = ctx
        self.qst_length = qst_length
        self.ctx_length = ctx_length
        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        self.lemmatizator = spacy.load('en_core_web_sm')

    def _preprocess_user_data(self, u_questions, u_contexts, qst_length, ctx_length, tokenizer, lemmatizator):
        """
           This function filters user input questions and contexts list from special symbols, lemmatizes and tokenizes them
           and extracts attention masks for these tokens, converts everything to numpy arrays and returns them
            Args:
             qst: list of user questions
             ctx: list of user contexts
             qst_length: variable that shows with how much tokens user questions data needs to be tokenized
             ctx_length: variable that shows with how much tokens user contexts data needs to be tokenized
             tokenizer: initiated tokenizer for tokenizing data
             lemmatizator: initiated lemmatizer for lemmatizing data

           Returns:
             Tokenized numpy array of user questions
             numpy array of attention mask of tokenized user questions
             Tokenized numpy array of user contexts
             numpy array of attention mask of tokenized user contexts

        """
        u_questions2 = copy.deepcopy(u_questions)
        u_contexts2 = copy.deepcopy(u_contexts)
        u_questions_filtered = [re.sub('[^a-zA-Z0-9]+', '', _.lower()) for _ in u_questions2]
        u_contexts_filtered = [re.sub('[^a-zA-Z0-9]+', '', _.lower()) for _ in u_contexts2]
        u_questions_lemmatized = []
        u_contexts_lemmatized = []
        for item in u_questions_filtered:
            stc = lemmatizator(item)
            stc_lemmatized = " ".join([token.lemma_ for token in stc])
            u_questions_lemmatized.append(stc_lemmatized)
        for item in u_contexts_filtered :
            stc = lemmatizator(item)
            stc_lemmatized = " ".join([token.lemma_ for token in stc])
            u_contexts_lemmatized .append(stc_lemmatized)
        qst_info = tokenizer.batch_encode_plus(u_questions_lemmatized, truncation=True, padding='max_length',
                                               return_attention_mask=True, max_length=qst_length)
        ctx_info = tokenizer.batch_encode_plus(u_contexts_lemmatized, truncation=True, padding='max_length',
                                               return_attention_mask=True, max_length=ctx_length)
        usr_qst_ids = qst_info['input_ids']
        usr_qst_mask = qst_info['attention_mask']
        usr_ctx_ids = ctx_info['input_ids']
        usr_ctx_mask = ctx_info['attention_mask']
        return np.array(usr_qst_ids, dtype=int), np.array(usr_qst_mask, dtype=int), np.array(usr_ctx_ids,
                                                                dtype=int), np.array(usr_ctx_mask, dtype=int)

    def predict_faq(self, u_questions, u_contexts, num_suggestions=5):
        """
         This function take lists of user questions and answers also a number of desired most suitable FAQ's, prepares
         user data calculates similarities between user inputs and FAQ's and prints results to command line
         Args:
           u_questions: list of user questions
           u_contexts: list of user contexts
           num_suggestions: desired number of most similar FAQ's to be returned for each user question. If nothing is
                provided 5 FAQ's will be returned for each question

         Returns:
           Prints all user questions and and given number of FAQ and FAQ answers for each of the questions

         """
        usr_qst_ids, usr_qst_mask, usr_ctx_ids, usr_ctx_mask = self._preprocess_user_data(u_questions, u_contexts,
                                                    self.qst_length, self.ctx_length, self.tokenizer, self.lemmatizator)
        question_mtrx = cosine_similarity(usr_qst_ids, self.qst_ids)
        context_mtrx = cosine_similarity(usr_ctx_ids, self.ctx_ids)
        # We assume that user context similarity is 5 times less important than user input similarity to original FAQ's
        final_mtrx = np.add(question_mtrx, context_mtrx * 0.2)
        best_answer_idxs = np.argsort(final_mtrx, axis=1)[:, -num_suggestions:][:, ::-1]
        for i in range(len(u_questions)):
            print('User input:', u_questions[i])
            print('These FAQ might be relevant for your question:')
            for j, (question, answer) in enumerate(zip([self.qst[j] for j in best_answer_idxs[i]],
                                                       [self.ans[j] for j in best_answer_idxs[i]])):
                print(f'{j + 1}. FAQ: {question}')
                print(f'   Answer: {answer}')


''' only for testing purposes
def make_contexts(lines):
    u_contexts = []
    for idx, _ in enumerate(lines):
        u_contexts.append('. '.join(lines[:idx]))
    return lines, u_contexts

'''


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True)) #exists=True
@click.argument('model_directory', type=click.Path())
def main(input_filepath, model_directory):
    """
        This function gets two arguments from command line, loads filtered JSON data file, lemmatizes questions and
        contexts  and then tokenizes them, then creates a model class and stores all information in this class that
        is required for mapping user questions to FAQ list. Creates a prediction method inside this class and serializes
        model class data to a file

        Args:
         input_filepath: command line argument which represents path where filtered json file should be located
          model_directory:  command line argument which represents directory where to store prepared model class
        Raises:
          If there is no two command line arguments when running this file or if there are no filtered input file in
          file system on given path error will be raised
    """
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    lemmatizator = spacy.load('en_core_web_sm')
    qst, ans, ctx = load_data_from_json(input_filepath)
    qst_ids, qst_mask, ctx_ids, ctx_mask = preprocess_data(qst, ctx, tokenizer, lemmatizator)
    print(model_directory)
    faq_prediction_model = PredictionModel(qst_ids, qst_mask, ctx_ids, ctx_mask, qst, ans, ctx, qst_ids.shape[1], ctx_ids.shape[1])
    with open(model_directory + '\\faq_prediction_model_' + str(date.today()) + '.pickle', 'wb') as f:
        pickle.dump(faq_prediction_model, f)
        print(f'Prediction model has been prepared and saved in {model_directory}')

    '''    only for testing purposes
      with open('..\..\data\external\input.txt') as f:
          lines = []
          for line in f:
              lines.append(line)
      u_questions, u_contexts = make_contexts(lines)
      faq_prediction_model.predict_faq(u_questions, u_contexts, num_suggestions=3)
      '''


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
