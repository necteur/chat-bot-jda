import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('punkt')


import numpy
import tflearn
import tensorflow
import random
import json
from tensorflow.python.framework import ops
with open("train.json") as file: #ouverture du fichier de données (dictionnaire python)
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:      #on parcourt tout le data (ici des dictionnaires de données) qui est compris dans le dictionnaire intents
    for pattern in intent["patterns"]:      # on parcourt le dictionnaire pour voir les différentes "feature(input que l'IA va devoir faire face)" appellé ici pattern
        wrds = nltk.word_tokenize(pattern)      # séparation des mots dans une phrase pour permettre la modularité
        words.extend(wrds)      #ajout de la liste wrds à la liste words pour connaître les mots qui sont dans la "feature"
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels: # ajout de tous les types de données pour qu'ils soient par la suite traités et que aucun ne soit oublié
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?"] #permet d'avoir la racine des mots et de comprendre le sens des mots ex :bnjr veux dire bonjour, gentiment ==> gentil ce qui va lui permettre de comprendre des mots dérivés (ex : "il est d'un gentille" l'IA va comprendre "il est gentil")
words = sorted(list(set(words))) # création d'une liste de mots simplifiés qui vont simplifier l'analyse des données

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

#création du réseau de neurones ici 4 neurones
ops.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])]) # couche d'entrer des neurones input
# couche cachée des neurones qui vont "réfléchir" pour déterminer les règles puis les utiliser
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # sortie des nerones avec "l'activation" qui va permettre de changé des nombres incompréhensibles en probabilité
#ici la fonction d'activation utilisée est softmax +info : https://fr.wikipedia.org/wiki/Fonction_softmax
net = tflearn.regression(net) # prédiction de la sortie à partir de l'entrée

model= tflearn.DNN(net)

#entrainement de l'IA : n_epch=x le nombre de fois que l'on va entrainer le bot
model.fit(training, output, n_epoch=10000, batch_size=16, show_metric=True)
model.save("model.tflearn")

#prédiction
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s) # on découpe les mot d'entrée
    s_words = [stemmer.stem(word.lower()) for word in s_words] # on prend les mots clefs

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
