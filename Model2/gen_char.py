from keras.models import model_from_json
import numpy as np

json_file = open('char_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("char_model.h5")


with open("austen-emma.txt", encoding='utf-8') as f:
    text = f.read().lower()
#print('corpus length:', len(text))
maxlen = 8

chars = sorted(list(set(text)))
#print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


start_index = np.random.randint(0, len(text) - maxlen - 1)
for diversity in [0.5]:
    #print('----- diversity:', diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    #print('----- Generating with seed: "' + sentence + '"')
    # print(generated)

    output = ''
    for i in range(100):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        # print(preds)
        #print(np.argmax(preds))
        next_index = sample(preds, diversity)
        #next_index = np.argmax(preds)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        output = output + next_char

    print(output)
       

