import csv, re, random, os, random, string
from os import listdir
from os.path import isfile, join
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction import text
from flask import Flask, request, jsonify
from textblob import TextBlob, Word


# Reading training data and doing cleanups
cat_train = list()
category = list()
stop = text.ENGLISH_STOP_WORDS
CORS = 'http://localhost:9000'


app = Flask(__name__)

@app.route('/predict', methods =['GET'])

def predict():
    
    # Prediction text
    text = request.args.get('text') if request.args.get('text') else ""
    # Split text using non alpha numeric, then remove stopwords
    text = ' '.join([i for i in re.split('[^a-z0-9]', text.lower()) if i not in stop])
    
    # Number of predictions
    predictions = int(request.args.get('num')) if request.args.get('num') else 3

    retval = dict()
    slicing_value = -1 * (predictions + 2)
    cat_train.insert(0, text)
    tfidf = TfidfVectorizer(max_df=0.4, min_df=2, stop_words='english', norm='l1',
                            use_idf=True).fit_transform(cat_train)
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[-2:slicing_value:-1]
    for index, x in enumerate(list(related_docs_indices)):
        cos_value = list(cosine_similarities)[x]
        if cos_value > 0:
            retval[category[x-1]] = str(cos_value)
    del cat_train[0]
    response = jsonify(retval)
    response.headers.add('Access-Control-Allow-Origin', CORS)
    #response.headers.add('Access-Control-Allow-Origin', 'https://dashboard.heroku.com/apps/bazarr-web')
    return response


@app.route('/post-tags', methods =['GET'])

def postTags():
    retval = {'tags':[]}
    text = re.sub('%20',' ', request.args.get('text')) if request.args.get('text') else ""

    # Remove characters that are not punctuations, numbers and alphabets
    text = re.sub('[^a-zA-Z0-9\s.?!-]','' ,text)
    # Remove extra spaces and tabs
    text = re.sub('[\s+]', ' ', text)

    print(text)

    blob = TextBlob(text.lower())
    temp = blob.tags

    for i in range(len(temp)):
        if temp[i][1] == 'JJ':
            k = 1
            found = ''
            while i+k < len(temp) and k < 2:
                if temp[i+k][1] == 'NN' or temp[i+k][1] == 'NNP':
                    if i+k+1 < len(temp) and (temp[i+k+1][1] == 'PRP'):
                        retval['tags'].append(string.capwords(temp[i][0] + ' ' + temp[i+k][0]) + ' ' + temp[i+k+1][0])
                    else:
                        retval['tags'].append(string.capwords(temp[i][0] + ' ' + temp[i+k][0]))
                    temp[i+k] = (temp[i+k][0],'DONE')
                    break
                k+=1

    for i in range(len(temp)):
        if temp[i][1] == 'NNP' or temp[i][1] == 'VBN' and temp[i][0] not in stop:
            retval['tags'].append(string.capwords(temp[i][0].lemmatize()))

    retval['tags'] = list(set(retval['tags']))
    response = jsonify(retval)
    response.headers.add('Access-Control-Allow-Origin', CORS)
    return (response)
        

def trainer(file):
    temp = ""
    word_count = 0
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index is not 1:
                row_words = re.sub("[^a-zA-Z -]", ' ', row[0].lower())
                if row_words not in stop:
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


