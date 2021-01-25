# import packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, \
    roc_curve, f1_score, auc
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import time


# split data and vectorize review text
def get_data(csv, vect, target):
    reviews = pd.read_csv(csv)

    reviews['Review'] = reviews['Review'].values.astype('U')
    Y = reviews[target]
    X = reviews.drop(['Recommended', 'Early Access'], axis=1)

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.30, random_state=42)

    xtrain_t = vect.fit_transform(xtrain.Review)
    xtrain_final = hstack([xtrain_t, csr_matrix(xtrain.Length).T], 'csr')

    xtest_t = vect.transform(xtest.Review)
    xtest_final = hstack([xtest_t, csr_matrix(xtest.Length).T], 'csr')

    return xtrain_final, xtest_final, ytrain, ytest


# tune hyper-parameters using cross-validation
def crossval(x, y):
    clf = XGBClassifier()
    param_grid = {
        'nthread': [4],
        'objective': ['binary:logistic'],
        'learning_rate': [0.05, 0.1, 0.5],
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2),
        'subsample': [0.8],
        'colsample_bytree': [0.7],
        'n_estimators': range(60, 220, 40),
        'use_label_encoder': [False],
        'eval_metric': ['logloss']
    }

    gsearch = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)

    gsearch.fit(x, y)

    res = gsearch.cv_results_
    params = gsearch.best_params_
    return res, params


def evaluate(real, pred):
    # compute auc score
    aucscore = roc_auc_score(real, pred)
    print("AUC Score: ", aucscore, "\n")

    # compute accuracy score
    accscore = accuracy_score(real, pred)
    print("Accuracy Score: ", accscore, "\n")

    # recall
    recall = recall_score(real, pred, average='weighted', zero_division=0)
    print("Recall: ", recall, "\n")

    # precision
    precision = precision_score(real, pred, average='weighted', zero_division=0)
    print("Precision: ", precision, "\n")

    # F1
    f1 = f1_score(real, pred, average='weighted', zero_division=0)
    print("F1: ", f1, "\n")

    # find values for confusion matrix
    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()
    print("--- Confusion Matrix--\n", tn, fp, fn, tp)

    array = [[tp, fp], [fn, tn]]

    df_cm = pd.DataFrame(array, index=["Pred Recommended", "Pred Not-Recommended"],
                         columns=["Act Recommended", "Act not-Recommended"])

    fig, ax = plt.subplots(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='')

    ax.set_title("XGBoost - Confusion Matrix\n")
    ax.xaxis.tick_top()
    plt.show()

    print("\nTrue Positives:", tp)
    print("False Positives ", fp)
    print("True Negatives: ", tn)
    print("False Negatives: ", fn)

    print("True Positive Rate: ", (tp / (tp + fn)))
    print("Specificity: ", (tn / (tn + fp)))
    print("False Positive Rate: ", (fp / (fp + tn)))

    fpr, tpr, threshold = roc_curve(real, pred)
    roc_auc = auc(fpr, tpr)

    # plot roc curve
    plt.title('XGBoost - ROC Curves')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return accscore, aucscore, recall, precision, f1


def get_results(acc, aucs, rec, prec, f1s, total_time):
    res = {'Accuracy': acc,
           'AUC Score': aucs,
           'Recall': rec,
           'Precision': prec,
           'F1': f1s,
           'Time Taken': total_time
           }

    res_df = pd.DataFrame([res], columns=['Accuracy', 'AUC Score', 'Recall', 'Precision', 'F1', 'Time Taken'])
    result = res_df.to_string()

    print(result, file=open('../results/XGBoost_Results.txt', 'w'))


# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=None, ngram_range=(1, 1), use_idf=True)

X_train, X_test, y_train, y_test = get_data('../data/transformed_reviews.csv', vectorizer, 'Recommended')

# cross-validation
results, best_params = crossval(X_train, y_train)

# final model
start_time = time.time()
model = XGBClassifier(nthread=best_params["nthread"], objective=best_params["objective"],
                      learning_rate=best_params["learning_rate"], max_depth=best_params["max_depth"],
                      min_child_weight=best_params["min_child_weight"], subsample=best_params["subsample"],
                      colsample_bytree=best_params["colsample_bytree"], n_estimators=best_params["n_estimators"],
                      use_label_encoder=best_params["use_label_encoder"], eval_metric=best_params["eval_metric"])

model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)
predictions = [round(value) for value in predictions]
end_time = time.time()
time_taken = end_time - start_time

# evaluate results
acc_score, auc_score, model_recall, model_precision, model_f1 = evaluate(y_test, predictions)

# print results to file
get_results(acc_score, auc_score, model_recall, model_precision, model_f1, time_taken)