import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

model = False
words = []
labels = []

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(str(s))
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def callthis(userq):
    train_the_model = False
    out = False
    global words
    global labels
    try:
        readfile = open("val.txt", "r")
        val = [readfile.readlines()]
        readfile.close()
        length = len(data['intents'])
        if length > int(val[0][0]):
            f = open("val.txt", "w")
            f.write(str(length))
            f.close()
            out = True
        else:
            with open("data.pickle", "rb") as f:
                words, labels, training, output = pickle.load(f)

        if out:
            rubbish
    except:
        train_the_model = True
        words = []
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

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

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
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    global model
    model = tflearn.DNN(net)

    try:
        if not train_the_model:
            model.load("model.tflearn")
        else:
            getout
    except:
        model.fit(training, output, n_epoch=500, batch_size=10, show_metric=True)
        model.save("model.tflearn")

    '''
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    '''
    ans = chat(userq)
    print(ans)
    return ans


def chat(userq):
    while True:
        inp = userq
        # if inp.lower() == "quit":
        #     break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        response = ""
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            response = str(random.choice(responses))
        else:
            response = str("Sorry, I didn't get your question. You can try again or ask other question")
        
        #print(response)
        return response


def bmi():
    mass = float(input('Enter your mass in kg: '))
    height = float(input('Enter your height in metres: '))
    bmi = float(mass) / (float(height) * float(height))

    sex = input("Enter your gender: ")

    print("Do you know your waist size ? It will help us in giving you better results. If yes then print 'yes' else print 'no'")
    ans = input()

    if ans == 'yes':
        while True:
            waist = int(input("Enter your waist size in inches"))
            if waist < 0 or waist > 100:
                print("Re-enter waist value")
                waist = int(input())
            else:
                break

    if bmi < 18.5:
        print(''''\033[1m'You are underweight!'\033[0m''')
        print(
            '\033[1m' + "Eat more frequently." + '\033[0m' + "When you're underweight, you may feel full faster. Eat five to six smaller meals during the day rather than two or three large meals." + '\n'
                                                                                                                                                                                                       '\033[1m' + "Choose nutrient-rich foods." + '\033[0m' + "As part of an overall healthy diet, choose whole-grain breads, pastas and cereals; fruits and vegetables; dairy products; lean protein sources; and nuts and seeds."
            + '\033[0m')
    elif 25 > bmi > 18.5:
        print("'\033[1m'You are Normal weight!'\033[0m'")
    elif bmi > 30:
        print("'\033[1m'You are Obese!'\033[0m'")
        # print("Eat more fruit, vegetables, nuts, and whole grains. + "
        # Exercise, even moderately, for at least 30 minutes a day.
        # Cut down your consumption of fatty and sugary foods.
        # Use vegetable-based oils rather than animal-based fats.")
    else:
        print("'\033[1m'You are over weight!'\033[0m'")
        # print('''Eat more fruit, vegetables, nuts, and whole grains.
        # Exercise, even moderately, for at least 30 minutes a day.
        # Cut down your consumption of fatty and sugary foods.
        # Use vegetable-based oils rather than animal-based fats.''')



