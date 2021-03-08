❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗ /!\ /!\ /!\ /!\ /!\ /!\ (☞ﾟヮﾟ)☞
LISEZ MOI LA C'EST IMPORTANT JE SAIS QUE TOUT LE MONDE ME ZAP (¬_¬ ) MAIS SANS LES INFORMATIONS QUE CONTIENENT LE DOCUMENT VOUS NE POURREZ PAS UTILISER LE CHATBOT MERCI
❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗ /!\ /!\ /!\ /!\ /!\ /!\
# chat-bot-jda
chat bot project NSI
but créer un chat bot qui pourra présenter le lycée JDA

documentation :
tuto tensorflow : https://www.youtube.com/watch?v=tPYj3fFJGjk
instalation de tensorflow : https://colab.research.google.com/drive/1F_EWVKa8rbMXi3_fG0w7AtcscFq7Hi7B#forceEdit=true&sandboxMode=true
documentation sur les résaux de neurones : https://colab.research.google.com/drive/1m2cg3D1x3j5vrFc-Cu0gMvc48gWyCOuG#forceEdit=true&sandboxMode=true

/!\ ATTENTION /!\
Ne pas utiliser python 3.9, versions prises en charge : 3.8(instable), 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0 (python 2.x, 1.x non pris en charge)
le programme ne fonctionne pas avec les librairies intégrées de base dans python
il est OBLIGATOIRE d'intaller :
  - tensorflow < pip install tensorflow > https://www.tensorflow.org/install?hl=fr
  - tflearn < pip install tflearn > http://tflearn.org/installation/
  - nltk < pip install nltk > https://www.nltk.org/install.html
  - pour la version non finie, l'intallation de tensorflow-gpu <pip install tensorflow-gpu> (qui permet de réaliser les entrainements de l'IA avec le GPU) est fortement conseillée. Mais pour cela il est nécessaire de posséder un GPU de la marque NVidia et d'installer CUDA Toolkit (qui peut nesséssiter la désinstallation en cas d'erreur de NVidia frameview SDK) lien de téléchargement : https://developer.nvidia.com/cuda-downloads?
  -  pyglet < pip install pyglet==1.3.2 >
Bon chat !!! ༼ つ ◕_◕ ༽つ 






Hi, I want to load the data for a chatbot that I have made before, but when I execute the program I have this error :

```Traceback (most recent call last):
  File "C:\Users\cm-nsi\Documents\GitHub\chat-bot\chatbot\fichier_de_lecture.py", line 34, in <module>
    model.load("model.tflearn")
  File "c:\programdata\anaconda3\lib\site-packages\tflearn\models\dnn.py", line 302, in load
    self.trainer.restore(model_file, weights_only, **optargs)
  File "c:\programdata\anaconda3\lib\site-packages\tflearn\helpers\trainer.py", line 500, in restore
    self.restorer.restore(self.session, model_file)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\training\saver.py", line 1326, in restore
    err, "a mismatch between the current graph and the graph")
tensorflow.python.framework.errors_impl.InvalidArgumentError: Restoring from checkpoint failed. This is most likely due to a mismatch between the current graph and the graph from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Assign requires shapes of both tensors to match. lhs shape= [8,8] rhs shape= [128,128]
	 [[node save_1/Assign_12 (defined at C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\framework\ops.py:1751) ]]

Original stack trace for 'save_1/Assign_12':
  File "C:\Program Files (x86)\pyzo\source\pyzo\pyzokernel\start.py", line 151, in <module>
    __pyzo__.run()
  File "C:\Program Files (x86)\pyzo\source\pyzo\pyzokernel\interpreter.py", line 222, in run
    self.guiApp.run(self.process_commands, self.sleeptime)
  File "C:\Program Files (x86)\pyzo\source\pyzo\pyzokernel\guiintegration.py", line 488, in run
    self._QtGui.real_QApplication.exec_()
  File "C:\Program Files (x86)\pyzo\source\pyzo\pyzokernel\interpreter.py", line 583, in process_commands
    self._process_commands()
  File "C:\Program Files (x86)\pyzo\source\pyzo\pyzokernel\interpreter.py", line 611, in _process_commands
    self.runfile(tmp)
  File "C:\Program Files (x86)\pyzo\source\pyzo\pyzokernel\interpreter.py", line 887, in runfile
    self.execcode(code)
  File "C:\Program Files (x86)\pyzo\source\pyzo\pyzokernel\interpreter.py", line 950, in execcode
    exec(code, self.locals)
  File "C:\Users\cm-nsi\Documents\GitHub\chat-bot\chatbot\fichier_de_lecture.py", line 31, in <module>
    model= tflearn.DNN(net)
  File "c:\programdata\anaconda3\lib\site-packages\tflearn\models\dnn.py", line 65, in __init__
    best_val_accuracy=best_val_accuracy)
  File "c:\programdata\anaconda3\lib\site-packages\tflearn\helpers\trainer.py", line 153, in __init__
    allow_empty=True)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\training\saver.py", line 828, in __init__
    self.build()
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\training\saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\training\saver.py", line 878, in _build
    build_restore=build_restore)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\training\saver.py", line 508, in _build_internal
    restore_sequentially, reshape)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\training\saver.py", line 350, in _AddRestoreOps
    assign_ops.append(saveable.restore(saveable_tensors, shapes))
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\training\saving\saveable_object_util.py", line 73, in restore
    self.op.get_shape().is_fully_defined())
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\ops\state_ops.py", line 227, in assign
    validate_shape=validate_shape)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\ops\gen_state_ops.py", line 65, in assign
    use_locking=use_locking, name=name)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\framework\op_def_library.py", line 793, in _apply_op_helper
    op_def=op_def)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\util\deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\framework\ops.py", line 3360, in create_op
    attrs, op_def, compute_device)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\framework\ops.py", line 3429, in _create_op_internal
    op_def=op_def)
  File "C:\Users\cm-nsi\AppData\Roaming\Python\Python37\site-packages\tensorflow_core\python\framework\ops.py", line 1751, in __init__
    self._traceback = tf_stack.extract_stack()
```

here is the program that I use to load the data :



```
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
from tensorflow.python.framework import ops

with open("train.json") as file:
    data = json.load(file)


with open("data.pickle", "rb") as f:
    words, labels, training, output =pickle.load(f)


ops.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model= tflearn.DNN(net)


model.load("model.tflearn")



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
```

and here is the program that I use to train the bot and create the checkpoint :

``` 
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
import pyglet
from tensorflow.python.framework import ops
with open("train.json", encoding="utf-8") as file: #ouverture du fichier de données (dictionnaire python)
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:      
    for pattern in intent["patterns"]:      
        wrds = nltk.word_tokenize(pattern)      
        words.extend(wrds)      
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels: 
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
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net) # prédiction de la sortie à partir de l'entrée

model= tflearn.DNN(net)

model.fit(training, output, n_epoch=40000, batch_size=64, show_metric=True)
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
```
Thanks for your help
