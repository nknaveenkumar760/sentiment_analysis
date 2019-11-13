import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
import collections
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import CountVectorizer


with open(r'Data/positive_sentiment.json') as dutch:
    data = json.load(dutch)

sentimentVal = []
contentVal = []
contentSentVal = []

for i in data:
    if i['sentiment'] == 'positive':

        sentiment = i['sentiment']
        content = i['content']
        contentVal.append(content)
        sentimentVal.append(sentiment)

# print(contentVal[1])

wordList = []
dictWords = {}
tokenizer = RegexpTokenizer(pattern="[^ ]+")
dutch_stops = set(stopwords.words('dutch'))

content_list = []
contentSentence = []
contentWord = []
flattened = []

# text = sentiment_content
for text in contentVal:

    # cleaning the text
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))      # remove punctuation like @,#,(),{} etc
    text = re.sub(r"\d+", "", text)          # to remove numarical values like 8,3,6,7 etc
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'\s*(?:https?://)?www\.\S*\.[A-Za-z]{2,5}\s*', '', text).strip()
    text = re.sub(r"RT\s+ ", "", text)
    text = re.sub(r"[_]\'", "\"", text)
    text = re.sub(r"\"", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"&.+;", "", text)
    text = re.sub("[a-zA-z]/", "", text)
    text = re.sub(r"\.", "", text)
    text = re.sub(r"rt", "", text)
    # text = re.sub(r"~", "\n", text)

    # print("clear text for tokenize-:", text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    content_list.append(text)

    sentTokens = sent_tokenize(text)
    contentSentence_list = [i + ".,1" for i in sentTokens if not i in dutch_stops]
    temp = [i.split(',') for i in contentSentence_list]
    contentSentence.append(temp)

    flattened = [val for sublist in contentSentence for val in sublist]

    wordTokens = word_tokenize(text)
    contentWord_list = [i for i in wordTokens if not i in dutch_stops]
contentWord.append(contentWord_list)


fdist = FreqDist(contentWord)

# print(len(flattened))
# print(flattened[4])
# print(contentWord_list[:5])
print(fdist.most_common(10))

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
vectorizer = CountVectorizer(lowercase=True, stop_words=dutch_stops, tokenizer=token.tokenize, ngram_range=(1, 3), min_df=2, max_df=1.0)
X = vectorizer.fit_transform(content_list[:10])
# print(vectorizer.get_feature_names())
# print(X.toarray())

y_val = [x[1] for x in fdist.most_common()]

fig = plt.figure(figsize=(10,5))
plt.plot(y_val)

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Positive)")
plt.show()

# fdist.plot(50,cumulative=True)
# plt.show()






