from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import json
import string
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model
import random
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

intents = json.loads(open('dialogbot.json').read())

inputs = []
tags = []
responses = {}
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', '<','>', '-', ':', ';']
words = []

for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])

data = pd.DataFrame({"inputs":inputs,
                    "tags":tags})

data["inputs"] = data["inputs"].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data["inputs"] = data["inputs"].apply(lambda wrd: ''.join(wrd))

tokenizer = Tokenizer(num_words = 2000)
tokenizer.fit_on_texts(data["inputs"])
train = tokenizer.texts_to_sequences(data["inputs"])

x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(data["tags"])
y_train


input_shape = x_train.shape[1]
print(input_shape)

unique_words = len(tokenizer.word_index)
output_length = le.classes_.shape[0]
print("unique words: ", unique_words)
print("output length: ", output_length)


classes = sorted(list(set(classes)))
print (len(classes), "classes", classes)


vocalbulary = len(tokenizer.word_index)
print("number of unique words : ", vocalbulary)

output_length = le.classes_.shape[0]
print("output lenghth", output_length)



from nltk.util import flatten
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense
from tensorflow.keras.models import Model
# Assuming you have defined input_range and output_length elsewhere in your code

i = Input(shape=(input_shape,))
x = Embedding(vocalbulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)
# Kompilasi model

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])


model = load_model('Models.h5')
print("model created")

def get_random_suggestion(num_patterns=5):
    # Gantilah logika untuk mendapatkan pola respons cepat
    all_patterns = []

    for _ in range(num_patterns):
        random_intent = random.choice(intents['intents'])
        random_pattern = random.choice(random_intent['patterns'])
        all_patterns.append(random_pattern)

    return all_patterns

def chatWithBot(user_input):
    textList = []
    prediction_input = []

    for letter in user_input:
        if letter not in string.punctuation:
            prediction_input.append(letter.lower())

    prediction_input = ''.join(prediction_input)
    textList.append(prediction_input)

    prediction_input = tokenizer.texts_to_sequences(textList)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], input_shape)

    output = model.predict(prediction_input)
    output = output.argmax()

    response_tag = le.inverse_transform([output])[0]
    response_options = responses.get(response_tag, [])  # Get the list of responses

    if response_options:
        return random.choice(response_options)
    else:
        # Jika tidak ada respons, berikan saran acak
        return get_random_suggestion()

def chat():
    while True:
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("BOT: Goodbye!")
            break

# Call the function to start the chat
#chat()
# save this as app.py

