# import packages
import pandas as pd

# read in csv
reviews = pd.read_csv('data/preprocessed_reviews.csv')

# add length column
reviews['Review'] = reviews['Review'].astype(str)
reviews['Length'] = reviews['Review'].apply(lambda x: len(x.split()))

# encode recommended & early access
encoded = pd.get_dummies(reviews, prefix=['Recommended', 'Early Access'],
                         columns=['Recommended', 'Early Access'], prefix_sep=" ")

encoded = encoded.drop(['Recommended False', 'Early Access False'], axis=1)

# rename columns
encoded = encoded.rename(columns={'Recommended True': 'Recommended', 'Early Access True': 'Early Access'})

# save to csv
encoded.to_csv('data/transformed_reviews.csv', index=False, encoding='utf-8')
