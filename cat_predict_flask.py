import csv, re, random, os, json, pickle
from os import listdir
from os.path import isfile, join
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from flask import Flask
from flask import request
app = Flask(__name__)


@app.route('/')
def predict():
    #print(stopwords.words("english"))
    # Remove stopwords from text in the future...
    
    # Prediction text
    text = request.args.get('text') if request.args.get('text') else ""
    # Number of predictions
    predictions = int(request.args.get('num')) if request.args.get('num') else 3

    retval = dict()
    slicing_value = -1 * (predictions + 2)
    cat_train.insert(0, text)
    tfidf = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english',
                            use_idf=True).fit_transform(cat_train)
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[-2:slicing_value:-1]
    for index, x in enumerate(list(related_docs_indices)):
        if list(cosine_similarities)[x] > 0:
            retval[category[x-1]] = list(cosine_similarities)[x]
    del cat_train[0]
    return json.dumps(retval)



cat_train = pickle.load(open("cat_train.p", "rb"))
category = pickle.load(open("category.p", "rb"))


if __name__ == '__main__':
    app.run(debug=False)


