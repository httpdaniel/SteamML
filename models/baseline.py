# import packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.dummy import DummyClassifier
import pandas as pd


# split data and vectorize review text
def get_data(csv, vect):
    reviews = pd.read_csv(csv)

    review_text = reviews['Review'].values.astype('U')
    recommended = reviews['Early Access']

    xtrain, xtest, ytrain, ytest = train_test_split(review_text, recommended, test_size=0.3, random_state=42)

    vect.fit(xtrain)
    xtrain = vect.transform(xtrain)

    return xtrain, xtest, ytrain, ytest


vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer(min_df=5, ngram_range=[1, 3])

X_train, X_test, y_train, y_test = get_data('../data/transformed_reviews.csv', vectorizer)

model = DummyClassifier(strategy='most_frequent')
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(vectorizer.transform(X_test))

# compute auc score
aucscore = roc_auc_score(y_test, predictions)
print("AUC Score: ", aucscore, "\n")

# compute accuracy score
accscore = accuracy_score(y_test, predictions)
print("Accuracy Score: ", accscore, "\n")

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print(pd.DataFrame(confusion_matrix(y_test, predictions),
                   columns=['Actual Recommended', "Actual Not-Recommended"],
                   index=['Predicted Recommended', 'Predicted Not-Recommended']))

print(f'\nTrue Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')

print(f'True Positive Rate: { (tp / (tp + fn))}')
print(f'Specificity: { (tn / (tn + fp))}')
print(f'False Positive Rate: { (fp / (fp + tn))}')
