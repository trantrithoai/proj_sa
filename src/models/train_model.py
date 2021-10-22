import argparse
import numpy as np
import re
import nltk
from typing import List
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Making the dataset.'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help='File to input',
    )
    parser.add_argument(
        '--output_model',
        type=str,
        required=True,
        help='File to output',
    )
    return parser.parse_args()


def read_training_file(path: str) -> List[str]:
    handler = open(path, 'r')
    for line in handler.readlines():
        yield line.strip()
    handler.close()


def main():
    args = parse_arguments()
    lines = read_training_file(args.train_file)
    documents = []
    labels = []
    for line in lines:
        parts = line.split('\t')
        documents.append(parts[0])
        labels.append(parts[1])

    tfidfconverter = TfidfVectorizer(
        max_features=1500,
        min_df=5,
        max_df=0.7,
        stop_words=stopwords.words('english')
    )
    tfidf_model = tfidfconverter.fit(documents)
    X = tfidfconverter.transform(documents).toarray()
    pickle.dump(tfidf_model, open("feature.pkl","wb"))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=0
    )
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)
    # Output the results
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    # Store the model
    with open(args.output_model, 'wb') as picklefile:
        pickle.dump(classifier,picklefile)


if __name__ == "__main__":
    main()
