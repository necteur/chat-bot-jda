import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('punkt')

from tkinter import *
import numpy
import tflearn
import tensorflow
import random
import json
import time
import os
from tensorflow.python.framework import ops
with open("train.json", encoding="utf-8") as file: #ouverture du fichier de données (dictionnaire python)
    data = json.load(file)

#prédiction
def traitement_des_donnees(s, features):
    bag = [0 for _ in range(len(features))]

    words_tok = nltk.word_tokenize(s) # on découpe les mot d'entrée
    words_tok = [stemmer.stem(word.lower()) for word in words_tok] # on prend les mots clefs

    for se in words_tok:
        for i, w in enumerate(features):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def post(a) :
        #print("Vous pouvez commencer à parler (taper quit pour arrêter)!")
        inp = input_user.get()
        messages.insert(INSERT, '%s\n' % inp)
        if inp.lower() == "quit":
            window.destroy()

        if inp.lower() == "bad apple":
            os.system("start cmd /k python ./ASCII_bad_apple-master/run.py")


        resultat = model.predict([traitement_des_donnees(inp, features)])
        resultat_index = numpy.argmax(resultat)
        tag = labels[resultat_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['reponses']
        aff_post = (random.choice(responses))
        print(aff_post)
        messages.insert(INSERT, '%s\n' % "Emma:")
        messages.insert(INSERT, '%s\n' % aff_post)
        messages.insert(INSERT, '\n')
        messages.insert(INSERT, '%s\n' % "vous: ")
        input_field.delete(0, 'end')


features = []
labels = []
mot1 = []
mot2 = []
training =[]
sortie = []

for intent in data["intents"]:      #on parcourt tout le data (ici des dictionnaires de données) qui est compris dans le dictionnaire intents
    for pattern in intent["patterns"]:      # on parcourt le dictionnaire pour voir les différentes "feature(input que l'IA va devoir faire face)" appellé ici pattern
        mots = nltk.word_tokenize(pattern)      # séparation des mots dans une phrase pour permettre la modularité
        features.extend(mots)      #ajout de la liste mots à la liste features pour connaître les mots qui sont dans la "feature"
        mot1.append(mots)
        mot2.append(intent["tag"])

        if intent["tag"] not in labels: # ajout de tous les types de données pour qu'ils soient par la suite traités et que aucun ne soit oublié
            labels.append(intent["tag"])

features = [stemmer.stem(w.lower()) for w in features if w not in "?"] #permet d'avoir la racine des mots et de comprendre le sens des mots ex :bnjr veux dire bonjour, gentiment ==> gentil ce qui va lui permettre de comprendre des mots dérivés (ex : "il est d'un gentille" l'IA va comprendre "il est gentil")
features = sorted(list(set(features))) # création d'une liste de mots simplifiés qui vont simplifier l'analyse des données

labels = sorted(labels)



out_empty = [0 for _ in range(len(labels))]

for x,doc in enumerate(mot1):
    bag = []

    mots= [stemmer.stem(w) for w in doc]

    for w in features:
        if w in mots:
            bag.append(1)
        else:
            bag.append(0)
    sortie_row = out_empty[:]
    sortie_row[labels.index(mot2[x])] = 1

    training.append(bag)
    sortie.append(sortie_row)


training = numpy.array(training)
sortie= numpy.array(sortie)

#création du réseau de neurones ici 4 neurones
ops.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])]) # couche d'entrer des neurones input
# couche cachée des neurones qui vont "réfléchir" pour déterminer les règles puis les utiliser (4 couche qui continnent 16 neurones)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, len(sortie[0]), activation="softmax") # sortie des nerones avec "l'activation" qui va permettre de changé des nombres incompréhensibles en probabilité
#ici la fonction d'activation utilisée est softmax +info : https://fr.wikipedia.org/wiki/Fonction_softmax
net = tflearn.regression(net) # prédiction de la sortie à partir de l'entrée

model= tflearn.DNN(net)

#entrainement de l'IA : n_epch=x le nombre de fois que l'on va entrainer le bot
model.fit(training, sortie, n_epoch=30, batch_size=128, show_metric=True)
model.save("model.tflearn")




#tkinter

window = Tk()
window.title("JDA chatbot : Emma")
messages = Text(window)
messages.pack()

input_user = StringVar()
input_field = Entry(window, text=input_user)
input_field.pack(side=BOTTOM, fill=X)

frame = Frame(window)
input_field.bind("<Return>", post)
messages.insert(INSERT, '%s\n' % "Vous pouvez commencer à parler (taper quit pour arrêter)")
messages.insert(INSERT, '\n')
messages.insert(INSERT, '%s\n' % "vous: ")
frame.pack()

window.mainloop()
