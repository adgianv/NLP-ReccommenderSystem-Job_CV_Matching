import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os
import statsmodels.api as sm
from IPython.display import display, HTML
from concurrent.futures import ThreadPoolExecutor
import sys
from collections import Counter
import seaborn as sns
import pickle

# NLP functions
from datasets import load_dataset
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import docx2txt
import requests
import subprocess
from tabulate import tabulate
# from textblob import TextBlob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from gensim import corpora, models
from gensim.matutils import Sparse2Corpus
from scipy.sparse import csr_matrix
from langdetect import detect
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
#!python -m spacy download en_core_web_sm
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
sp = spacy.load('en_core_web_sm')

porter=SnowballStemmer("english")
lmtzr = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

import warnings
# Ignore SettingWithCopyWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # Set norms to 1 if they are 0 to avoid division by zero
    if norm_vec1 == 0:
        norm_vec1 = 1
    if norm_vec2 == 0:
        norm_vec2 = 1
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

# Additional functions
def strip(word):
    mod_string = re.sub(r'\W+', '', word)
    return mod_string

def abbr_or_lower(word):
    if re.match('([A-Z]+[a-z]*){2,}', word):
        return word
    else:
        return word.lower()
    
def tokenize(text, modulation):
    if modulation < 2:
        tokens = re.split(r'\W+', text)
        stems = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            lowers= token.lower()
            if lowers not in stop_words:
                if re.search('[a-zA-Z]', lowers):
                    if modulation==0:
                        stems.append(lowers)
                    if modulation==1:
                        stems.append(porter.stem(lowers))
    else:
        sp_text=sp(text)
        stems = []
        lemmatized_text=[]
        for word in sp_text:
            lemmatized_text.append(word.lemma_)
        stems = [abbr_or_lower(strip(w)) for w in lemmatized_text if (abbr_or_lower(strip(w))) and (abbr_or_lower(strip(w)) not in stop_words)]
    return " ".join(stems)

def vectorize(tokens, vocab):
    vector=[]
    for w in vocab:
        vector.append(tokens.count(w))
    return vector

def remove_special_characters(input_string):
    # Keep only alphabetic characters (a-z, A-Z)
    cleaned_string = re.sub(r'[^a-zA-Z]', ' ', input_string)
    cleaned_string = ' '.join(cleaned_string.split())
    return cleaned_string

def preprocess_job_titles(df):
    # Remove some of the most occurring words from the job titles
    exclude_words = ["senior", "lead", "director","graduate","remote","madrid","lima","barcelona","administrator","applied","entry","liaison","curation","finnish","est","hours","preferred","product","months","month","manager","valencia","head","sevilla","privacy","siri","intern","gij","sr","associate","new","time","needs","need","problem","junior","canada","week","temporary","mexico","rotation","planning","america","asia","regional","resident","bogota","day","small","spain","disney","snowflake","january","iam","czech","public","private","sports","tumor","tumors","sport","september"]
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(
            word if word.lower() not in exclude_words else ''
            for word in title.split()
        )
    )
    # Remove the name of the company from the job titles
    df['cleaned_job_title'] = df.apply(
        lambda row: ' '.join(
            word if word.lower() not in row['companies'].lower() else ''
            for word in row['cleaned_job_title'].split()
        ),
        axis=1
    )
    # Remove 'data' if 'business' is present in the job titles
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(
            word if word.lower() != 'data' or 'business' not in title.lower() else ''
            for word in title.split()
        )
    )
    # Replace 'science' with 'scientist' in the job titles
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(
            word if word.lower() != 'science' else 'scientist'
            for word in title.split()
        )
    )
    # Replace 'analitics','analystics', with 'analyst' in the job titles
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(
            word if word.lower() not in ['analitics', 'analystics','analytics'] else 'analyst'
            for word in title.split()
        )
    )
    # Replace 'engineering' with 'engineer' in the job titles
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(
            word if word.lower() != 'engineering' else 'engineer'
            for word in title.split()
        )
    )
    # Replace 'machine learning' with 'ml' in the job titles
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(
            word if word.lower() not in ['machine', 'learning'] else 'ml'
            for word in title.split()
        )
    )
    # Replace 'artificial intelligence' with 'ai' in the job titles
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(
            word if word.lower() not in ['artificial', 'intelligence'] else 'ai'
            for word in title.split()
        )
    )
    # Removes the characters such as (f/m/d) from the job titles
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(
            word if (len(word) >= 2 or word.lower() in ["ai", "ml", "it", "db", "qa", "bi", "r", "c"]) else ''
            for word in title.split()
        )
    )
    # Remove duplicated words from the job titles (e.g. 'data data scientist' -> 'data scientist')
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(
            word for word in set(title.split())
        )
    )

    # Tokenize and preprocess job titles using CountVectorizer
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'), lowercase=True)
    X = vectorizer.fit_transform(df['cleaned_job_title'])
    word_frequencies = X.sum(axis=0)

    # Threshold for least occurring words (we are mainly interested in the most occurring words such as data, software, engineer, etc.)
    # Threshold of 60% has been chosen to reduce both the cadinality and the number of empty strings in the job titles
    threshold = int(0.8 * len(vectorizer.get_feature_names_out()))

    # Get the indices of the most occurring words
    most_occurring_indices = word_frequencies.argsort()[0, -threshold:]
    most_occurring_words = vectorizer.get_feature_names_out()[most_occurring_indices]

    # Get the indices of least occurring words
    least_occurring_indices = word_frequencies.argsort()[0, :threshold]
    least_occurring_words = vectorizer.get_feature_names_out()[least_occurring_indices]
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: ' '.join(word for word in title.split() if word not in least_occurring_words)
    )
    # Fill empty strings with Other
    print("Number of empty strings in cleaned_job_title: ", df['cleaned_job_title'].apply(lambda x: x.strip()).eq('').sum(), " -> \'Other\'")
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda x: x.strip() if x.strip() else 'Other'
    )
    # Replace the job titles with the simplified ones
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: 'Data_Scientist' if 'data' in title and 'scientist' in title else title
    )
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: 'Business' if 'business' in title or 'analyst' in title or "bi" in title else title
    )
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: 'Developer' if 'developer' in title or 'python' in title or 'java' in title or 'developement' in title or 'software' in title or 'it' in title else title
    )
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: 'Researcher' if 'testing' in title or 'research' in title or 'phd' in title else title
    )
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: 'Database' if 'database' in title or 'sql' in title or 'db' in title else title
    )
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: 'ML_Engineer' if 'engineer' in title or 'security' in title or 'network' in title or 'ml' in title or 'ai' in title else title
    )
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: 'Data_Scientist' if 'data' in title  else title
    )
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: 'Business' if 'manager' in title or 'managment' in title else title
    )
    # Other for the rest of the job titles
    df['cleaned_job_title'] = df['cleaned_job_title'].apply(
        lambda title: 'Other' if title not in ['Data_Scientist', 'Business', 'Developer', 'ML_Engineer', 'Database', 'Researcher'] else title
    )
    return df
    


class Metrics:
    def __init__(self):
        self.results = {}
        self.colors = {}

    def run(self, y_true, y_pred, method_name, color='blue', average='macro'):
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)

        # Store results
        self.results[method_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        self.colors[method_name] = color

    def plot(self):
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))

        # Plot each metric
        for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
            ax = axs[i//2, i%2]
            values = [res[metric] * 100 for res in self.results.values()]
            colors = [self.colors[method] for method in self.results.keys()]
            ax.bar(self.results.keys(), values, color=colors)
            ax.set_title(metric)
            ax.set_ylim(0, 100)

            # Add values on the bars
            for j, v in enumerate(values):
                ax.text(j, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
            
            # Adjust layout
            ax.set_xticklabels(self.results.keys(), rotation=45)
   
        plt.tight_layout()
        plt.show()

    def display(self):
        # Display results and values
        display(HTML(tabulate(self.results, headers='keys', tablefmt='html')))
        display(HTML(tabulate(self.results.values(), headers=self.results.values()[0].keys(), tablefmt='html')))