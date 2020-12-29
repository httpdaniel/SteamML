# import packages

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk
import re
import string

# read in data
reviews = pd.read_csv('data/parsed_reviews.csv')


# convert nltk tags to wordnet tags
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# remove numbers, punctuation, and make lowercase
def clean_text(text):
    no_numbers = re.sub(r'\d+', '', text)
    no_punctuation = "".join([char.lower() for char in no_numbers if char not in string.punctuation])
    return no_punctuation


# perform tokenization, stopword removal and lemmatization
def process_text(text):
    text = clean_text(text)
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []

    for word in word_tokens:
        if word not in stop_words:
            filtered_sentence.append(word)

    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(filtered_sentence)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmas = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmas.append(word)
        else:
            lemmas.append(lemmatizer.lemmatize(word, tag))

    sentence = " ".join(lemmas)
    return sentence


# Pre-process review text and replace
reviews['Review'] = reviews['Review'].apply(process_text)
reviews.to_csv('data/preprocessed_reviews.csv', index=False, encoding='utf-8')
