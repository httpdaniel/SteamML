# import packages

import json_lines
import pandas as pd
import json

# parse JSON file

parsed_reviews = []
with open('data/steam_reviews.jl', 'rb') as f:
    for item in json_lines.reader(f):
        review = {
            'Review': item['text'],
            'Recommended': item['voted_up'],
            'Early Access': item['early_access']
        }
        parsed_reviews.append(review)

reviews = json.dumps(parsed_reviews, indent=4)
df = pd.read_json(reviews)
df.to_csv('data/reviews.csv', index=False, encoding='utf-8')

