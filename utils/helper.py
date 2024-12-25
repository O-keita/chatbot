import torch
import numpy as np

from data.preprocess import tokenize, bag_of_words


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def chat_response(msg, model, intents, all_words, tags):

    sentence = tokenize(msg)

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float().to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:

                return np.random.choice(intent['responses'])
            
    else:
       return "Sorry, I didn't have answer to  that."

