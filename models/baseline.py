# import packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


# split data and vectorize review text
def get_data(csv, vect, feature, target):
    reviews = pd.read_csv(csv)

    review_text = reviews[feature].values.astype('U')
    recommended = reviews[target]

    xtrain, xtest, ytrain, ytest = train_test_split(review_text, recommended, test_size=0.3, random_state=42)

    vect.fit(xtrain)
    xtrain = vect.transform(xtrain)

    return xtrain, xtest, ytrain, ytest


def evaluate(real, pred):
    # compute auc score
    aucscore = roc_auc_score(real, pred)
    print("AUC Score: ", aucscore, "\n")

    # compute accuracy score
    accscore = accuracy_score(real, pred)
    print("Accuracy Score: ", accscore, "\n")

    # recall
    recall = recall_score(real, pred, average=None)
    print("Recall: ", recall, "\n")

    # precision
    precision = precision_score(real, pred, average=None, zero_division=0)
    print("Precision: ", precision, "\n")

    # find values for confusion matrix
    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()
    print("--- Confusion Matrix--\n", tn, fp, fn, tp)

    array = [[tp, fp], [fn, tn]]

    df_cm = pd.DataFrame(array, index=["Pred Recommended", "Pred Not-Recommended"],
                         columns=["Act Recommended", "Act not-Recommended"])

    fig, ax = plt.subplots(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='')

    ax.set_title("Baseline - Confusion Matrix\n")
    ax.xaxis.tick_top()
    plt.show()

    print("\nTrue Positives:", tp)
    print("False Positives ", fp)
    print("True Negatives: ", tn)
    print("False Negatives: ", fn)

    print("True Positive Rate: ", (tp / (tp + fn)))
    print("Specificity: ", (tn / (tn + fp)))
    print("False Positive Rate: ", (fp / (fp + tn)))


vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer(min_df=5, ngram_range=[1, 3])

X_train, X_test, y_train, y_test = get_data('../data/transformed_reviews.csv', vectorizer, 'Review', 'Recommended')

# model = DummyClassifier(strategy='most_frequent')
model = DummyClassifier(strategy='uniform')
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(vectorizer.transform(X_test))

# evaluate results
evaluate(y_test, predictions)
