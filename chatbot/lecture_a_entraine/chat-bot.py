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
with open("../train.json", encoding="utf-8") as file: #ouverture du fichier de données (dictionnaire python)
    data = json.load(file)



## résaux de neurones
def res_neurones(training) :
    ops.reset_default_graph()
    resaux_neurones = tflearn.input_data(shape=[None, len(training[0])]) # couche d'entrer des neurones input + récupération des donné pour l'entrainement
    # couche cachée des neurones qui vont "réfléchir" pour déterminer les règles puis les utiliser (4 couche qui continnent 128 neurones chaqune). Chaque neurone est connecter a tout les neurones de la couche précédente et de la couche suivante (d'où le fully_connected
    resaux_neurones = tflearn.fully_connected(resaux_neurones, 128)
    resaux_neurones = tflearn.fully_connected(resaux_neurones, 128)
    resaux_neurones = tflearn.fully_connected(resaux_neurones, 128)
    resaux_neurones = tflearn.fully_connected(resaux_neurones, 128)
    resaux_neurones = tflearn.fully_connected(resaux_neurones, len(sortie[0]), activation="softmax") # sortie des nerones avec "l'activation" qui va permettre de changé des nombres incompréhensibles en probabilité
    #ici la fonction d'activation utilisée est softmax +info : https://fr.wikipedia.org/wiki/Fonction_softmax
    resaux_neurones = tflearn.regression(resaux_neurones) # prédiction de la sortie à partir de l'entrée

    model= tflearn.DNN(resaux_neurones) #définition du model du résaux de neurones
    return model


##prédiction
def traitement_des_donnees(s, features):
    bag = [0 for _ in range(len(features))]

    words_tok = nltk.word_tokenize(s) # on retire la ponctuation
    words_tok = [stemmer.stem(word.lower()) for word in words_tok] # on prend les mots clefs en retirant tout les mots courts et on racourcie les mots à leurs racines pour une meilleur compréenssion pour le chatbot

    for j in words_tok:
        for i, k in enumerate(features):
            if k == j:
                bag[i] = 1 #on regarde si les mot qui sont dans l'entrer sont dans notre features donc dans nos questions de base proposez à la machine, si oui on rajoute de la probabilité à la réponse qui corespond
                print(bag)

    return numpy.array(bag) # on transforme nos résultat sous forme de probabilité
## traitement des entrés lors de la discution
def post(a) :
        #print("Vous pouvez commencer à parler (taper quit pour arrêter)!")
        inp = input_user.get()
        messages.insert(INSERT, '%s\n' % inp) #insertion du message
        if inp.lower() == "quit":
            window.destroy()

        if inp.lower() == "bad apple": # lance le programme qui permet de lancer dans une console une vidéo en ASCII
            os.system("start cmd /k python ../ASCII_bad_apple-master/run.py")
            messages.insert(INSERT, '%s\n' % "Emma:")
            messages.insert(INSERT, '%s\n' % "programme creer par Chion82 integre par Antoine Cateux et Rihan Chaudhory")
            messages.insert(INSERT, '\n')
            messages.insert(INSERT, '%s\n' % "vous: ")
            input_field.delete(0, 'end')

        else :

            resultat = model.predict([traitement_des_donnees(inp, features)])[0] # retourne une probabilité de toute les réponses prédéfinies soient la bonne
            resultat_index = numpy.argmax(resultat) # nous renvoie l'index de la plus grande valeur dans notre liste de probalité des réponses prédéfinie
            tag = labels[resultat_index]
            if resultat[resultat_index] > 0.65 : # on regarde si la plus grande probabilité est supérieur à 0.65 pour savoir si la réponse est fiable
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['reponses']
                aff_post = (random.choice(responses))
                #print(aff_post)
                messages.insert(INSERT, '%s\n' % "Emma:")
                messages.insert(INSERT, '%s\n' % aff_post)
                messages.insert(INSERT, '\n')
                messages.insert(INSERT, '%s\n' % "vous: ")
                input_field.delete(0, 'end')
            else :
                messages.insert(INSERT, '%s\n' % "Emma:")
                messages.insert(INSERT, '%s\n' % "je n'ai pas compris pourriez vous reformuler votre question.")
                messages.insert(INSERT, '\n')
                messages.insert(INSERT, '%s\n' % "vous: ")
                input_field.delete(0, 'end')



## lecture du dictionnaire
features = []
labels = []
mot1 = []
mot2 = []
training =[]
sortie = []

for intent in data["intents"]:      #on parcourt tout le data (ici des dictionnaires qui contienne toutes les données) qui est compris dans le dictionnaire intents
    for pattern in intent["patterns"]:      # on parcourt le dictionnaire pour voir les différentes "feature(entré que l'IA va devoir faire face)" appellé ici pattern
        mots = nltk.word_tokenize(pattern)      # séparation des mots et supression de la ponctuation (accent, virgule) dans la phrase
        features.extend(mots)      #ajout de la liste mots à la liste features pour connaître les mots qui sont dans la partie "pattern" du dictionnaire
        mot1.append(mots)
        mot2.append(intent["tag"])

        if intent["tag"] not in labels: # ajout de tous les types de données pour qu'ils soient par la suite traités et que aucun ne soit oublié
            labels.append(intent["tag"])


## traitement des donnés
features = [stemmer.stem(i.lower()) for i in features if i not in "?"] #permet d'avoir la racine des mots et de comprendre le sens des mots ex :bnjr veux dire bonjour, gentiment ==> gentil ce qui va lui permettre de comprendre des mots dérivés (ex : "il est d'un gentille" l'IA va comprendre "il est gentil"). On retire aussi les points d'intérogation
features = sorted(list(set(features))) # création d'une liste de mots simplifiés qui vont simplifier l'analyse des données

labels = sorted(labels) # on organise les donnés



sortie_vide = [0 for _ in range(len(labels))] # on fait une liste remplit de 0, que l'on remplira par la suite par des 1
print(sortie_vide)

for i,doc in enumerate(mot1):
    bag = []

    mots= [stemmer.stem(w) for w in doc] # on prend les racines des mots de nos questions prédéfinie

    for w in features:
        if w in mots:
            bag.append(1)
        else:
            bag.append(0)
    sortie_row = sortie_vide[:] #on copie la liste dans une autre liste
    sortie_row[labels.index(mot2[i])] = 1 # on remplace les 0 par des 1


    training.append(bag)
    sortie.append(sortie_row)

# on rend les sorties interprétables pour le résaux de neurones
training = numpy.array(training)
sortie= numpy.array(sortie)





##entrainement
#entrainement de l'IA : n_epch=x le nombre de fois que l'on va entrainer le bot, batch_size la quantité de donner que l'on donne a chaque entrainement, show_metric=True permet de montrer ce qu'il se passe pour obtenir les information tel que la précision du Chatbot
model=res_neurones(training)
model.fit(training, sortie, n_epoch=2000, batch_size=150, show_metric=True)
model.save("model.tflearn") ## on enregistre les donnés




## affichage tkinter

window = Tk() #création de la fenêtre
window.title("JDA chatbot : Emma")
messages = Text(window)
messages.pack()

#entré pour parler avec le chat bot
input_user = StringVar()
input_field = Entry(window, text=input_user)
input_field.pack(side=BOTTOM, fill=X)

#fenêtre où tout ce que l'on va entré et toutes les réponses vont s'afficher
frame = Frame(window)
input_field.bind("<Return>", post)
messages.insert(INSERT, '%s\n' % "Vous pouvez commencer à parler (taper quit pour arrêter)")
messages.insert(INSERT, '\n')
messages.insert(INSERT, '%s\n' % "vous: ")
frame.pack()

window.mainloop()
