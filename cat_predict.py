import csv, re, random, os
import pickle
from os import listdir
from os.path import isfile, join
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords


cat_train = list()
category = list()

def trainer(file):
    temp = ""
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index is not 1:
                row_words = re.sub("[^a-zA-Z -]", ' ', row[0].lower())
                temp += ' ' + str(row_words)
    temp = re.sub('\s+', ' ', temp)
    cat_train.append(temp)


def predict(text, predictions=1):
    
    retval = list()
    slicing_value = -1 * (predictions + 2)
    cat_train.insert(0, text)
    tfidf = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english',
                             use_idf=True).fit_transform(cat_train)
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[-2:slicing_value:-1]
    for index, x in enumerate(list(related_docs_indices)):
        if list(cosine_similarities)[x] > 0:
            retval.append([category[x-1], list(cosine_similarities)[x]])
    del cat_train[0]
    return retval


category_training_path = os.getcwd() + "/Category_Training_Data/"
files = [f for f in listdir(category_training_path) if isfile(join(category_training_path, f))]
for file in files:
    if file != '.DS_Store':
        category.append(re.sub('.csv', '', file))
        trainer(category_training_path + file)

pickle.dump(cat_train, open("cat_train.p", "wb"))
pickle.dump(category, open("category.p", "wb"))

