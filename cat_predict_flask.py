import csv, re, random, os, json
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
    predictions = request.args.get('num') if request.args.get('num') else 3

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


category_training_path = os.getcwd() + "/Category_Training_Data/"
files = [f for f in listdir(category_training_path) if isfile(join(category_training_path, f))]
for file in files:
    if file != '.DS_Store':
        category.append(re.sub('.csv', '', file))
        trainer(category_training_path + file)


if __name__ == '__main__':
    app.run(debug=True)


