import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
nltk.download('punkt')

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from tkinter import *
from tensorflow.python.framework import ops

with open("train.json") as file:
    data = json.load(file)

with open("data.pickle", "rb") as f:
    words, labels, training, output =pickle.load(f)

ops.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model= tflearn.DNN(net)


model.load("./model.tflearn")

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


def post(a) :
        #print("Vous pouvez commencer à parler (taper quit pour arrêter)!")
        inp = input_user.get()
        messages.insert(INSERT, '%s\n' % "vous: ")
        messages.insert(INSERT, '%s\n' % inp)
        if inp.lower() == "quit":
            window.destroy()

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['reponses']
        aff_post = (random.choice(responses))
        print(aff_post)
        messages.insert(INSERT, '%s\n' % "Emma:")
        messages.insert(INSERT, '%s\n' % aff_post)
        messages.insert(INSERT, '\n')
        input_field.delete(0, 'end')


#tkinter

window = Tk()
window.title("best chatbot ever")
messages = Text(window)
messages.pack()

input_user = StringVar()
input_field = Entry(window, text=input_user)
input_field.pack(side=BOTTOM, fill=X)



frame = Frame(window)
input_field.bind("<Return>", post)
messages.insert(INSERT, '%s\n' % "Vous pouvez commencer à parler (taper quit pour arrêter)")
messages.insert(INSERT, '\n')
frame.pack()

window.mainloop()