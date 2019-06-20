
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import tflearn
import random
import pickle
import json


class Predict:

    ERROR_THRESHOLD = 0.25
    rawRootLink = "data/"
    trainingRootLink = "models/"
    context = {}

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)

        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=False):
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)

        return (np.array(bag))

    def classify(self, sentence):
        results = self.model.predict([self.bow(sentence, self.words)])[0]
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], r[1]))
        return return_list

    def __init__(self, training_data_file_name, raw_data_file_name, tflearn_logs, tflearn_model):
        self.training_data_file_name = self.trainingRootLink + training_data_file_name
        self.raw_data_file_name = self.rawRootLink + raw_data_file_name
        self.tflearn_logs = tflearn_logs
        self.tflearn_model = self.trainingRootLink + tflearn_model

        self.stemmer = LancasterStemmer()

        training_file = open(self.training_data_file_name, "rb")
        data = pickle.load(training_file)
        self.words = data['words']
        self.classes = data['classes']
        self.train_x = data['train_x']
        self.train_y = data['train_y']

        with open(self.raw_data_file_name, encoding='utf-8') as json_data:
            self.intents = json.load(json_data)

        tf.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(self.train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.train_y[0]), activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

        self.model = tflearn.DNN(net, tensorboard_dir=self.tflearn_logs)

        self.bow('Tôi muốn mua hàng công nghệ', self.words)

        self.model.load('./' + self.tflearn_model)

    def response(self, sentence, userID='1', show_details=False):
        results = self.classify(sentence)
        if results:
            while results:
                for i in self.intents['intents']:
                    if i['tag'] == results[0][0]:
                        if 'context_set' in i:
                            if show_details:
                                ('context:', i['context_set'])
                            self.context[userID] = i['context_set']
                        if not 'context_filter' in i or (
                                userID in self.context and 'context_filter' in i and i['context_filter'] == self.context[userID]):
                            if show_details:
                                print('tag:', i['tag'])
                            return random.choice(i['responses'])

                results.pop(0)