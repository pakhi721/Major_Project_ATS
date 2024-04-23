import os
import re
import pickle
import string
import unicodedata
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords
from wordcloud import STOPWORDS, WordCloud

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed

import nltk
nltk.download('stopwords')

from nltk.tokenize import word_tokenize

#nltk.download('punkt')
num_epochs = 10  # Or any other value you choose

def preprocess_text(article):
    if isinstance(article, str):  # Check if the input is a string
        article = article.lower()
        article = re.sub(r'<.*?>', '', article)
        article = re.sub(r'[^a-zA-Z\s]', '', article)
        article = word_tokenize(article)
        stop_words = set(stopwords.words('english'))
        article = [word for word in article if word not in stop_words]
        stemmer = SnowballStemmer('english')
        #article = [stemmer.stem(word) for word in article]
        return ' '.join(article)
    else:
        return ''  # Return an empty string for non-string values


# Load the dataset
df = pd.read_csv('cleaned_data.csv')

# Preprocess the articles and highlights
df['article'] = df['article'].apply(preprocess_text)
df['highlights'] = df['highlights'].apply(preprocess_text)

# Add start and end tokens to the highlights
df['highlights'] = df['highlights'].apply(lambda x: f'_START_ {x} _END_')

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize the articles and highlights
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(train_df['article'])

y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(train_df['highlights'])

# Pad sequences
max_article_len = 500
max_highlight_len = 20

x_train = pad_sequences(x_tokenizer.texts_to_sequences(train_df['article']), maxlen=max_article_len, padding='post')
y_train = pad_sequences(y_tokenizer.texts_to_sequences(train_df['highlights']), maxlen=max_highlight_len, padding='post')

x_val = pad_sequences(x_tokenizer.texts_to_sequences(val_df['article']), maxlen=max_article_len, padding='post')
y_val = pad_sequences(y_tokenizer.texts_to_sequences(val_df['highlights']), maxlen=max_highlight_len, padding='post')

# Define the model architecture
latent_dim = 240
embedding_dim = 300

encoder_input = Input(shape=(max_article_len,))
encoder_embedding = Embedding(len(x_tokenizer.word_index) + 1, embedding_dim, input_length=max_article_len, trainable=False)(encoder_input)
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_embedding)
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output, state_h, state_c = encoder_lstm2(encoder_output1)

decoder_input = Input(shape=(None,))
decoder_embedding = Embedding(len(y_tokenizer.word_index) + 1, embedding_dim, trainable=True)(decoder_input)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = TimeDistributed(Dense(len(y_tokenizer.word_index) + 1, activation='softmax'))
decoder_output = decoder_dense(decoder_output)

model = Model([encoder_input, decoder_input], decoder_output)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([x_train, y_train[:, :-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:], epochs=num_epochs, batch_size=64, validation_data=([x_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))

# Save the tokenizers
with open('x_tokenizer.pkl', 'wb') as f:
    pickle.dump(x_tokenizer, f)

with open('y_tokenizer.pkl', 'wb') as f:
    pickle.dump(y_tokenizer, f)

# Save the model
model.save('lstm_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Maximum length for input sequence
max_article_len = 43

app = Flask(__name__)



def generate_summary(text):
    # Generate summary for the input text
    preprocessed_text = preprocess_text(text)
    predictions = model.predict(preprocessed_text)
    # Decode the predicted summary
    decoded_summary = decode_sequence(predictions[0], tokenizer)
    return decoded_summary

def decode_sequence(input_seq, tokenizer):
    # Decode the predicted sequence into text
    decoded_sentence = ''
    for token_index in input_seq:
        if token_index == 0:
            break
        word = tokenizer.index_word.get(token_index, '')
        if word:
            decoded_sentence += ' ' + word
    return decoded_sentence.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_summary', methods=['POST'])
def summarize():
    # Get user input from the form
    input_text = request.form['input_text']
    # Generate summary
    summary = generate_summary(input_text)
    return render_template('summary.html', input_text=input_text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
