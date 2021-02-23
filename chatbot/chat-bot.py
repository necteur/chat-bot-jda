import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()



import numpy
import tflearn
import tensorflow
import random
import json
from tensorflow.python.framework import ops
with open("train.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]: #on parcour tout le data
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern) # séparation des mot dans une phrases pour permettre la modularité
        words.extend(wrds) #ajout de la liste wrds à la liste words
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels: # ajout de tout les types de donné pour qu'il soit par la suite traité
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training =[]
output = []

out_empty = [0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
    bag = []

    wrds= [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output= numpy.array(output)

ops.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model= tflearn.DNN(net)

model.fit(training, output, n_epoch=5000, batch_size=8, show_metric=True)
model.save("model.tflearn")

#prédiction
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Vous pouvez commencer à parler (taper quit pour arrêter)!")
    while True:
        inp = input("Vous: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['reponses']

        print(random.choice(responses))

chat()
