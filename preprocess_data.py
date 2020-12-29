# import packages

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

# read in data
reviews = pd.read_csv('data/reviews.csv')


def process_text(text):
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []

    for word in word_tokens:
        if word not in stop_words:
            filtered_sentence.append(word)

    sentence = " ".join(filtered_sentence)
    return sentence


reviews['Review'] = reviews['Review'].apply(process_text)
reviews.to_csv('data/preprocessed_reviews.csv', index=False, encoding='utf-8')
