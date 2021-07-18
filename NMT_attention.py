# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 00:25:34 2021

@author: User
"""
import pandas as pd
import os
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GRU, Dense 

"""## Loading the dataset using tensorflow_datasets """
dataset=pd.read_csv('data.txt',header=None,sep='\t',encoding='utf-8-sig')

NUM_SAMPLES = len(dataset)
"""## Adding start and end sequence markers"""

en_sentences = []
cn_sentences = []
for i in tqdm(range(NUM_SAMPLES)):
    en_sentences.append(dataset[0][i])
    cn_sentences.append('startseq ' + dataset[1][i] + ' endseq')

"""## Tokenization"""

en_tk = Tokenizer(num_words=5000)
cn_tk = Tokenizer(num_words=5000)

with open('en.pkl', 'wb') as f:
    pickle.dump(en_tk, f)

with open('cn.pkl', 'wb') as f:
    pickle.dump(cn_tk, f)
    
en_tk.fit_on_texts(en_sentences)
cn_tk.fit_on_texts(cn_sentences)

en_lens = [len(x.split()) for x in en_sentences]
cn_lens = [len(x.split()) for x in cn_sentences]
en_max = np.max(en_lens)
cn_max = np.max(cn_lens)

max_sequence_len = max(en_max,cn_max)
en_vocab_size = 5000
cn_vocab_size =5000

"""## Tokenization and sequence padding"""

def preprocess_en(en):
    encoded_en = en_tk.texts_to_sequences(en)
    padded_en = pad_sequences(encoded_en, maxlen=max_sequence_len, padding='post', truncating='post')
    return padded_en

def preprocess_cn(cn):
    encoded_cn = cn_tk.texts_to_sequences(cn)
    padded_cn = pad_sequences(encoded_cn, maxlen=max_sequence_len, padding='post', truncating='post')
    return padded_cn

def preprocess_text(en, cn):
    return preprocess_en(en.numpy().decode()), preprocess_cn(cn.numpy().decode())

"""## Preparing training dataset"""

train_en = preprocess_en(en_sentences)
train_cn = preprocess_cn(cn_sentences)

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((train_en, train_cn))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(1024)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

"""## Defininf Encoder, Decoder and Attention models using tf.keras model subclassing"""

class Encoder(tf.keras.Model):
    def __init__(self, hidden_size=1024, max_sequence_len=30, batch_size=batch_size, embedding_dim=256, vocab_size=5000):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.GRU_1 = GRU(units=hidden_size, return_sequences=True)
        self.GRU_2 = GRU(units=hidden_size,
                         return_sequences=True, return_state=True)

    def initial_hidden_state(self):
        return tf.zeros(shape=(self.batch_size, self.hidden_size))

    def call(self, x, initial_state, training=False):
        x = self.embedding_layer(x)
        x = self.GRU_1(x, initial_state=initial_state)
        x, hidden_state = self.GRU_2(x)
        return x, hidden_state

"""
Attention:Calculate the context vector belonging to each timestep
"""

class Attention(tf.keras.Model):
    def __init__(self, hidden_size=256):
        super(Attention, self).__init__()
        self.fc1 = Dense(units=hidden_size)
        self.fc2 = Dense(units=hidden_size)
        self.fc3 = Dense(units=1)

    def call(self, encoder_output, hidden_state, training=False):
        '''hidden_state : h(t-1)'''
        y_hidden_state = tf.expand_dims(hidden_state, axis=1)
        y_hidden_state = self.fc1(y_hidden_state)
        y_enc_out = self.fc2(encoder_output)
        
        #get a_ij
        y = tf.keras.backend.tanh(y_enc_out + y_hidden_state)
        attention_score = self.fc3(y)
        attention_weights = tf.keras.backend.softmax(attention_score, axis=1)
        
        #get c_i
        context_vector = tf.multiply(encoder_output, attention_weights)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, hidden_size=1024, max_sequence_len=30, batch_size=batch_size, embedding_dim=256, vocab_size=5000):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
    
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.GRU = GRU(units=hidden_size,
                       return_sequences=True, return_state=True)
        self.attention = Attention(hidden_size=self.hidden_size)
        self.fc = Dense(units=self.vocab_size)

    def initial_hidden_state(self):
        return tf.zeros(shape=(self.batch_size, self.hidden_size))

    def call(self, x, encoder_output, hidden_state, training=False):
        x = self.embedding_layer(x)
        context_vector, attention_weights = self.attention(
            encoder_output, hidden_state, training=training)
        contect_vector = tf.expand_dims(context_vector, axis=1)
        x = tf.concat([x, contect_vector], axis=-1)
        x, curr_hidden_state = self.GRU(x)
        x = tf.reshape(x, shape=[self.batch_size, -1])
        x = self.fc(x)
        return x, curr_hidden_state, attention_weights

"""## Defining training loop, loss function and optimizer"""

loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
train_accuracy = tf.metrics.SparseCategoricalAccuracy()


def loss_function(y_true, y_pred):
    loss = loss_object(y_true, y_pred)
    mask = 1 - tf.cast(tf.equal(y_true, 0), 'float32')
    return tf.reduce_mean(loss * mask)
    
@tf.function()
def training_step(en, cn):    
    with tf.GradientTape() as Tape:
        encoder_init_state = encoder.initial_hidden_state()
        encoder_output, encoder_hidden_state = encoder(en, encoder_init_state, training=True)
        decoder_hidden = encoder_hidden_state
        loss = 0
        acc = []
        current_word = tf.expand_dims(cn[:, 0], axis=1)
        for word_idx in range(1, max_sequence_len):
            next_word = cn[:, word_idx]
            logits, decoder_hidden, attention_weights = decoder(current_word, encoder_output, decoder_hidden, training=True)
            loss += loss_function(next_word, logits)
            acc.append(train_accuracy(next_word, logits))
            current_word = tf.expand_dims(next_word, axis=1)
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = Tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss, tf.reduce_mean(acc)

encoder = Encoder()
decoder = Decoder()
checkpoint_dir = 'training'
checkpoint_prefix = 'training/ckpt'
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

"""## Training the model"""

epochs = 4
num_steps = NUM_SAMPLES // batch_size
for epoch in range(1, epochs + 1):
    print(f'Epoch {epoch}/{epochs}')
    ep_loss = []
    ep_acc = []
    progbar = tf.keras.utils.Progbar(target=num_steps, stateful_metrics=[
                                     'curr_loss', 'curr_accuracy'], unit_name='batch')

    for step, example in enumerate(train_dataset):
        en = example[0]
        cn = example[1]
        loss, acc = training_step(en, cn)
        loss /= cn.shape[1]
        ep_loss.append(loss)
        ep_acc.append(acc)
        progbar.update(
            step + 1, values=[('curr_loss', loss), ('curr_accuracy', acc)])

    if epoch % 1 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    print(f'Metrics after epoch {epoch} : Loss => {np.mean(ep_loss):.3f} | Accuracy => {np.mean(ep_acc):.3f}')
checkpoint.save(file_prefix=checkpoint_prefix)

"""## Inference function"""

def translate_sentence(sentence):
    print(sentence)
    sentence = preprocess_en([sentence])
    enc_init = tf.zeros(shape=[1, 1024])
    enc_out, enc_hidden = encoder(sentence, enc_init)

    decoder.batch_size = 1
    cn_tk.index_word[0] = ''
    decoded = []
    att = []
    current_word = tf.expand_dims([cn_tk.word_index['startseq']], axis=0) 
    decoder_hidden = enc_hidden
    for word_idx in range(1, max_sequence_len):
        logits, decoder_hidden, attention_weights = decoder(current_word, enc_out, decoder_hidden)
        decoded_idx = np.argmax(logits)
        if cn_tk.index_word[decoded_idx] == 'endseq':
            break
        decoded.append(cn_tk.index_word[decoded_idx])
        
        att.append(attention_weights.numpy().squeeze())
        current_word = tf.expand_dims([decoded_idx], axis=0)
    return ' '.join(decoded), att

"""## Translating and visualizing Attention maps"""

sentences = ['Is disability insurance required by law ?','Can creditors take life insurance after death ?','Does travelers insurance have renters insurance ?']

for inp_sentence in sentences:
    inp_array = inp_sentence.split()
    inp_len = len(inp_sentence.split())
    trans_sentence, attention_weights = translate_sentence(inp_sentence)
    #print("",attention_weights)
    trans_array = trans_sentence.split()
    trans_len = len(trans_array)
    attention_weights = np.array([x for x in attention_weights])
    
    attention_weights = attention_weights[:trans_len,:inp_len]
    plt.figure(figsize=(inp_len, trans_len))
    plt.matshow(attention_weights)
    plt.xticks(range(inp_len), inp_array, rotation=90)
    plt.yticks(range(trans_len), trans_array)
    print('EN : ', inp_sentence)
    print('CN : ', trans_sentence.replace(' ',''))
    print('-'*30)
