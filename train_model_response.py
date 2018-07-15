import pickle
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# Restore all of your data structures.
data = pickle.load(open("training_data","rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intent file
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Load our saved tensorflow model.
model.load('./model.tflearn')

# Clean up the sentence.
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # Stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence,words,show_details=False):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if(w==s):
                bag[i] = 1
                if(show_details):
                    print("found in bag : %s"%w)
    return(np.array(bag))

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # Generate probabilities from the model.
    results = model.predict([bow(sentence,words)])[0]
    # Filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x:x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]],r[1]))
    # Return tuple of intent and probability
    return return_list

def response(sentence,userID='123',show_details=False):
    results = classify(sentence)
    # If we have a classification then find the matching intent tag
    if results:
        # Loop as long as there are matches to process.
        while results:
            for i in intents['intents']:
                # Find a tag matching the first result.
                if(i['tag']==results[0][0]):
                    # Set context for this intent if necessary.
                    if('context_set' in i):
                        if show_details:
                            print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # Check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                       (userID in context and 'context_filter' in i and i['context_filter']==context[userID]):
                        if show_details:
                            print('Tag : ',i['tag'])
                        # A random response from intent
                        return (random.choice(i['responses']))
            results.pop(0)

def main():
    print("--------------HERE IS THE NEW CHATBOT CREATED---------------")
    print("--------------START TALKING--------------")
    print("Press 0 anytime if you want to exit the chat.")
    print()
    print("User :",end=" ")
    inp = input()
    while inp!="0":
        bot_reply = response(inp)
        print("Bot : ",bot_reply)
        print("User :",end=" ")
        inp = input()
    print()
    print("----------------BYE! VISIT AGAIN---------------")

if(__name__=="__main__"):
    main()







    
