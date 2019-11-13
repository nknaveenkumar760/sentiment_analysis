import json
import re
import string


with open(r'Data/negative_sentiment.json') as file:
    data = json.load(file)

# with open(r'Data/pos_sentiment_3.json') as file:
#     data += json.load(file)

with open(r'Data/neg_sentiment_3.json') as file:
    data += json.load(file)

with open(r'Data/pos_sentiment_4.json') as file:
     data += json.load(file)

with open(r'Data/neg_sentiment_4.json') as file:
    data += json.load(file)

# with open(r'Data/positive_sentiment.json') as file:
#     data = json.load(file)


fileCreate = open(r"sentiment_data.txt", "w+")
for i in data:
    text = i['content']
    senti = i['sentiment']
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation like @,#,(),{} etc
    text = re.sub(r"\d+", "", text)  # to remove numarical values like 8,3,6,7 etc
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'\s*(?:https?://)?www\.\S*\.[A-Za-z]{2,5}\s*', '', text).strip()
    text = re.sub(r"RT\s+ ", "", text)
    text = re.sub(r"[_]\'", "\"", text)
    text = re.sub(r"\"", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"&.+;", "", text)
    text = re.sub("[a-zA-z]/", "", text)
    text = re.sub(r"\.", "", text)
    text = re.sub(r"»", "", text)
    text = re.sub(r"’", "", text)
    text = re.sub(r"rt", "", text)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r"\n", "", text)

    senti = senti.replace("positive", "1")
    senti = senti.replace("negative", "0")
    # print(text)

    fileCreate.write(text + "," + senti + "\n")

with open("newfile1.txt", "r") as file:
    data = file.readlines()
for i in data:
    fileCreate.write(i)

fileCreate.close()