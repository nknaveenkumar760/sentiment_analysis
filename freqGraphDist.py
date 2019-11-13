from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

with open("neg_sentiment_data.txt", "r") as text_file:
    data = text_file.read().split('\n')

dutch_stops = set(stopwords.words('dutch'))



tokenizer = RegexpTokenizer(r'\w+')
tokens = []
contentList = []

for i in data:
    i = re.sub(r"0", "", i)
    i = re.sub(r"1", "", i)
    i = re.sub(r"rt", "", i)

    contentList.append(i)

    wordTokens = tokenizer.tokenize(i)
    contentWord_list = [i for i in wordTokens if not i in dutch_stops]
    tokens.extend(contentWord_list)

print(contentList[4])
# print(contentList[2])
fdist = FreqDist(tokens).most_common(100)
# print(fdist)
# fdist.plot(20, title='Word Frequency Distribution (Positive)')
# plt.show()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(contentList[:5])
colName = vectorizer.get_feature_names()
matrix = X.toarray()

df = pd.DataFrame(matrix, columns=colName)
print(df)