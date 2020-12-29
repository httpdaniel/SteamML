# import packages
import pandas as pd
import matplotlib.pyplot as plt

# read in csv
reviews = pd.read_csv('../data/transformed_reviews.csv')

recommended = reviews[reviews['Recommended'] == 1]
not_recommended = reviews[reviews['Recommended'] == 0]

# Plot histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
ax1.hist(recommended['Length'].values, bins=6)
ax1.set(xlabel='Word count of \'Recommended\' reviews', ylabel='Frequency')
ax2.hist(not_recommended['Length'].values, bins=6)
ax2.set(xlabel='Word count of \'Not Recommended\' reviews', ylabel='Frequency')
plt.show()
