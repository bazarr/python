import csv, re, random, os, random, string
import pickle
from os import listdir
from os.path import isfile, join
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from textblob import TextBlob, Word

stop = text.ENGLISH_STOP_WORDS

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

def postTags():
    with open('post.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    
    content = ' '.join(content)
    content = re.sub('[^a-zA-Z0-9\s.?!-]','' ,content)
    content = re.sub('[\s+]', ' ', content)

    blob = TextBlob(content.lower())

    temp = blob.tags
    val = []
    for i in range(len(temp)):
        if temp[i][1] == 'JJ':
            k = 1
            found = ''
            while i+k < len(temp) and k < 2:
                if temp[i+k][1] == 'NN' or temp[i+k][1] == 'NNP':
                    
                    if i+k+1 < len(temp) and (temp[i+k+1][1] == 'PRP'):
                        val.append(string.capwords(temp[i][0] + ' ' + temp[i+k][0]) + ' ' + temp[i+k+1][0])
                    else:
                        val.append(string.capwords(temp[i][0] + ' ' + temp[i+k][0]))
                    temp[i+k] = (temp[i+k][0],'DONE')
                    break
                k+=1

    for i in range(len(temp)):
        if temp[i][1] == 'NNP' or temp[i][1] == 'VBN' and temp[i][0] not in stop:
            val.append(string.capwords(temp[i][0].lemmatize()))

    print(val)



category_training_path = os.getcwd() + "/Category_Training_Data/"
files = [f for f in listdir(category_training_path) if isfile(join(category_training_path, f))]
for file in files:
    if file != '.DS_Store':
        category.append(re.sub('.csv', '', file))
        trainer(category_training_path + file)

pickle.dump(cat_train, open("cat_train.p", "wb"))
pickle.dump(category, open("category.p", "wb"))

