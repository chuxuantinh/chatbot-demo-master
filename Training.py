import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle


class Training:

    inputRootLink = "data/"
    outputRootLink = "models/"

    def __init__(self, input_file_name, out_put_file_name, tflearn_logs, tflearn_model):
        self.input_file_name = self.inputRootLink + input_file_name
        self.out_put_file_name = self.outputRootLink + out_put_file_name
        self.tflearn_logs = tflearn_logs
        self.tflearn_model = self.outputRootLink + tflearn_model

    def ngrams(self, str, n):
        tokens = str.split(' ')
        arr = []
        for i in range(len(tokens)):
            new_str = ''
            if i == 0 and n > 1:
                new_str = '_'
                for j in range(n):
                    if j < n - 1:
                        if (i + j) <= len(tokens):
                            new_str += ' ' + tokens[i + j]
                        else:
                            new_str += ' _'
            else:
                for j in range(n):
                    if j < n:
                        if (i + j) < len(tokens):
                            if j == 0:
                                new_str += tokens[i + j]
                            else:
                                new_str += ' ' + tokens[i + j]
                        else:
                            new_str += ' _'
            arr.append(new_str)
        return arr

    def execute(self):
        stemmer = LancasterStemmer()

        with open(self.input_file_name, encoding='utf-8') as json_data:
            intents = json.load(json_data)

        self.ngrams('a b c d e f g h', 4)
        words = []
        classes = []
        documents = []
        ignore_words = ['?', 'và', 'à', 'ừ', 'ạ', 'vì', 'từng', 'một_cách']

        for intent in intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                words.extend(w)
                documents.append((w, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
        words = sorted(list(set(words)))

        classes = sorted(list(set(classes)))

        print(len(documents), "documents")
        print(len(classes), "classes", classes)
        print(len(words), "unique stemmed words", words)

        training = []
        output = []

        output_empty = [0] * len(classes)

        for doc in documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        print(train_x[1])
        print(train_y[1])

        tf.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

        model = tflearn.DNN(net, tensorboard_dir=self.tflearn_logs)

        model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
        model.save(self.tflearn_model)

        file = open(self.out_put_file_name, mode="wb")
        pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, file)