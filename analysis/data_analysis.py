# import packages
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt


# read in csv
reviews = pd.read_csv('../data/transformed_reviews.csv')

recommended = reviews[reviews['Recommended'] == 1]
not_recommended = reviews[reviews['Recommended'] == 0]

vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=['game', 'oyun', 'monika', 'que', 'show', 'juego',
                                                             'spiel', 'jogo', 'игра', 'игрy', 'jeu', 'как']
                             ).fit(reviews['Review'].values.astype('U'))


# plot histograms - word count
def word_count(rec, not_rec):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    plt.title('Word Count')
    ax1.hist(rec['Length'].values, bins=6)
    ax1.set(xlabel='Word count of \'Recommended\' reviews', ylabel='Frequency')
    ax2.hist(not_rec['Length'].values, bins=6)
    ax2.set(xlabel='Word count of \'Not Recommended\' reviews', ylabel='Frequency')
    plt.show()


# find most common bi-grams
def bigram_common(text):

    words = vectorizer.transform(text)
    sum_words = words.sum(axis=0)

    ngram_count = []

    for word, index in vectorizer.vocabulary_.items():
        ngram_count.append((word, sum_words[0, index]))

    sorted_ngram_count = sorted(ngram_count, key=lambda z: z[1], reverse=True)
    sorted_ngram_count = sorted_ngram_count[:20]

    x = []
    y = []

    for i in sorted_ngram_count:
        x.append(i[0])
        y.append(i[1])

    fig = plt.figure(figsize=(20, 20))
    plt.barh(x, y)
    fig.gca().invert_yaxis()
    plt.title('Most Common Bi-grams')
    plt.xlabel('Frequency')
    plt.ylabel('Words')

    fig.show()


word_count(recommended, not_recommended)
bigram_common(recommended['Review'].values.astype('U'))
bigram_common(not_recommended['Review'].values.astype('U'))
