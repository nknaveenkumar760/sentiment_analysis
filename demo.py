from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('TkAgg')
import pandas
import random


def get_all_data():

    with open("sentiment_data.txt", "r") as text_file:
        data = text_file.read().split('\n')
    random.seed(1022)
    random.shuffle(data)
    return data

def preprocessing_data(data):
    processing_data = []
    for single_data in data:
        if len(single_data.split(",")) == 2 and single_data.split(",")[1] != "":
            processing_data.append(single_data.split(","))

    return processing_data

def split_data(data):
    total = len(data)
    training_ratio = 0.75
    training_data = []
    evaluation_data = []

    for indice in range(0, total):
        if indice < total * training_ratio:
            training_data.append(data[indice])
        else:
            evaluation_data.append(data[indice])

    return training_data, evaluation_data

def preprocessing_step():
    data = get_all_data()
    processing_data = preprocessing_data(data)

    return split_data(processing_data)

def training_step(data, vectorizer):
    training_text = [data[0] for data in data]
    training_result = [data[1] for data in data]

    training_text = vectorizer.fit_transform(training_text)
    return BernoulliNB().fit(training_text, training_result)

# Count Vectorizer
data =get_all_data()

training_data, evaluation_data = preprocessing_step()
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data[:5]) # slicing is done to fit the dataframe into the output screen
colName = vectorizer.get_feature_names()
matrix = x.toarray()

df = pandas.DataFrame(matrix, columns=colName)
print('---------------------------Count Vectorization Table---------------------------\n\n',df,"\n")

classifier = training_step(training_data, vectorizer)

def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))


def print_result(result):
    text, analysis_result = result
    print_text = "Positive" if analysis_result[0] == '1' else "Negative"
    print(text, ":", print_text)

def simple_evaluation(evaluation_data):

    evaluation_text     = [evaluation_data[0] for evaluation_data in evaluation_data]
    evaluation_result   = [evaluation_data[1] for evaluation_data in evaluation_data]

    total = len(evaluation_text)
    corrects = 0
    for index in range(0, total):
        analysis_result = analyse_text(classifier, vectorizer, evaluation_text[index])
        text, result = analysis_result
        corrects += 1 if result[0] == evaluation_result[index] else 0

    return corrects * 100 / total

def create_confusion_matrix(evaluation_data):
    evaluation_text = [evaluation_data[0] for evaluation_data in evaluation_data]
    actual_result = [evaluation_data[1] for evaluation_data in evaluation_data]
    prediction_result = []
    for text in evaluation_text:
        analysis_result = analyse_text(classifier, vectorizer, text)
        prediction_result.append(analysis_result[1][0])

    matrix = confusion_matrix(actual_result, prediction_result)
    return matrix


confusion_matrix_result = create_confusion_matrix(evaluation_data)

# Put your testing text here --->
print("--------------------------------Testing-------------------------------\n")
print_result( analyse_text(classifier, vectorizer,"areaal pootgoed krimpt"))
print_result( analyse_text(classifier, vectorizer,"xin als mijn dochter klaar is met aardbeien planten"))
print_result( analyse_text(classifier, vectorizer,"martijnvdam we willen geen ggo maïs bt   of mon op onze akkers hoe gaat u stemmen in de raad"))
print('\n')

true_negatives = confusion_matrix_result[0][0]
false_negatives = confusion_matrix_result[0][1]
false_positives = confusion_matrix_result[1][0]
true_positives = confusion_matrix_result[1][1]

accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) * 100
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print('---------------------------Model Performance---------------------------\n')
print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)


