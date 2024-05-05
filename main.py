import os
import yaml

import pandas as pd
import numpy as np
import random as rn
import re
import pickle
# import numpy as np
# import random as rn
# import re
# import pickle
# from tqdm import tqdm
#
# import matplotlib.pyplot as plt
#
from keras.layers import Embedding, LSTM, Dense, Softmax
from keras.models import Model
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences





import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import tensorflow as tf
from keras.layers import Bidirectional, Concatenate
MAXLEN = 15

QSN_VOCAB_SIZE = 16
ANS_VOCAB_SIZE = 38

LSTM_UNITS = 128
BATCH_SIZE = 4
EMBEDDING_SIZE = 300
dir_path = 'raw_data'
# files_list = os.listdir(dir_path + os.sep)

questions = list()
answers = list()
unique_tokens = set()

stream = open( 'sample.yaml' , 'rb')
docs = yaml.safe_load(stream)
conversations = docs['conversations']
for con in conversations:
        if len( con ) > 2 :
            questions.append(con[0])
            replies = con[ 1 : ]
            ans = ''
            for rep in replies:
                ans += ' ' + rep
            answers.append( ans )
        elif len( con )> 1:
            questions.append(con[0])
            answers.append(con[1])
# for filepath in files_list:
answers_with_tags = list()
for i in range(len(answers)):
    if type(answers[i]) == str:
        answers_with_tags.append(answers[i])
    else:
        questions.pop(i)

answers = list()
for i in range(len(answers_with_tags)):
    answers.append('<START> ' + answers_with_tags[i] + ' <END>')
    print(answers)

from keras_preprocessing import text
enc_tokenizer = Tokenizer(filters='', oov_token='<unk>')
enc_tokenizer.fit_on_texts(questions)
sequences = np.array(list(enc_tokenizer.word_index.items()))
print(sequences)
# # sequences1 = np.array(list(enc_tokenizer.word_index.items()))
dec_tokenizer = Tokenizer(filters='', oov_token='<unk>')
dec_tokenizer.fit_on_texts(answers)
sequences1 = np.array(list(dec_tokenizer.word_index.items()))
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1
import re
def tokenize(sentences):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append(tokens)
    return tokens_list, vocabulary
from keras import layers, activations, models, preprocessing
import numpy as np
import keras
from keras import preprocessing, utils
tokenized_questions = enc_tokenizer.texts_to_sequences(questions)
# print(tokenized_questions)
maxlen_questions = max(len(x) for x in tokenized_questions)
padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')
encoder_input_data = np.array(padded_questions)
tokenized_answers = dec_tokenizer.texts_to_sequences(answers)
# print(tokenized_answers)
maxlen_answers = max(len(x) for x in tokenized_answers)
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
decoder_input_data = np.array(padded_answers)
tokenized_answers = dec_tokenizer.texts_to_sequences(answers)
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]

padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
# onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)
decoder_output_data = np.array(padded_answers)
print(decoder_output_data.shape)

class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence
    returns encoder-outputs, encoder_final_state_h, encoder_final_state_c
    '''

    def __init__(self, inp_vocab_size, embedding_size, lstm_size, input_length):
        super().__init__()
        self.vocab_size = inp_vocab_size
        self.embedding_dim = embedding_size
        self.lstm_units = lstm_size
        self.input_length = input_length
        self.enc_output = self.enc_state_h = self.enc_state_c = 0

        # Initialize Embedding layer, output shape: (batch_size, input_length, embedding_dim)
        # self.embedding = Embedding(self.vocab_size, self.embedding_dim,
        #                            embeddings_initializer=tf.keras.initializers.Constant(enc_embedding_matrix),
        #                            trainable=False,
        #                            input_length=self.input_length, mask_zero=True, name="encoder_Embedding")
        self.embedding = Embedding(self.vocab_size, self.embedding_dim,
                                   trainable=True, mask_zero=True,
                                   name="encoder_Embedding")

        # Intialize Encoder LSTM layer
        self.lstm = LSTM(units=self.lstm_units, activation='tanh', recurrent_activation='sigmoid',
                         kernel_initializer=tf.keras.initializers.glorot_uniform(seed=26),
                         recurrent_initializer=tf.keras.initializers.orthogonal(seed=54),
                         bias_initializer=tf.keras.initializers.zeros(),
                         return_state=True, return_sequences=True, name="encoder_LSTM")


        self.lstm = LSTM(units=self.lstm_units, activation='tanh', recurrent_activation='sigmoid',
                         kernel_initializer=tf.keras.initializers.glorot_uniform(seed=26),
                         recurrent_initializer=tf.keras.initializers.orthogonal(seed=54),
                         bias_initializer=tf.keras.initializers.zeros(),
                         return_state=True, return_sequences=True, name="encoder_LSTM")

# Bidirectional layer
        self.bidirectional = Bidirectional(self.lstm)


    def call(self, input_sequence, states):
        '''
          This function takes a sequence input and the initial states of the encoder.
        '''
        # Embedding inputs, using pretrained glove vectors
        # shape: (input_length, glove vector's dimension)
        embedded_input = self.embedding(input_sequence)
        # print('encoder input')
        # print(embedded_input)
        # mask for padding
        mask = self.embedding.compute_mask(input_sequence)


        self.enc_out, enc_fw_state_h, enc_bw_state_h, enc_fw_state_c, enc_bw_state_c = self.bidirectional(embedded_input, mask=mask)

        # Concatenating forward and backward states
        # enc_state_h and c shape: (batch_size, 2*lstm_size)
        self.enc_state_h = Concatenate()([enc_fw_state_h, enc_bw_state_h])
        self.enc_state_c = Concatenate()([enc_fw_state_c, enc_bw_state_c])
        return self.enc_out, self.enc_state_h, self.enc_state_c, mask

    def initialize_states(self,batch_size):
        '''
        Given a batch size it will return intial hidden state and intial cell state.
        '''
        # shape: batch_size,input_length,lstm_size
        return (tf.zeros([batch_size, 2*self.lstm_units]), tf.zeros([batch_size, 2*self.lstm_units]))
vocab_size=10
embedding_size=20
lstm_size=32
input_length=10
batch_size=16
encoder=Encoder(vocab_size,embedding_size,lstm_size,input_length)
input_sequence=tf.random.uniform(shape=[batch_size,input_length],maxval=vocab_size,minval=0,dtype=tf.int32)
initial_state=encoder.initialize_states(batch_size)
# print(input_sequence)
encoder_output,state_h,state_c, enc_mask=encoder(input_sequence,initial_state)

# print(encoder_output.shape, state_h.shape, state_c.shape, enc_mask.shape)

class Attention(tf.keras.layers.Layer):
    '''
    Class the calculates score based on the scoring_function using Bahdanu attention mechanism.
    '''

    def __init__(self, scoring_function, att_units):
        super().__init__()
        self.scoring_function = scoring_function
        self.att_units = att_units

        # Initializing for 3 kind of losses
        if self.scoring_function == 'dot':
            # Intialize variables needed for Dot score function here
            self.dot = tf.keras.layers.Dot(axes=[2, 2])
        if scoring_function == 'general':
            # Intialize variables needed for General score function here
            self.wa = Dense(self.att_units)
        elif scoring_function == 'concat':
            # Intialize variables needed for Concat score function here
            self.wa = Dense(self.att_units, activation='tanh')
            self.va = Dense(1)

    def call(self, decoder_hidden_state, encoder_output, enc_mask):
        '''
        Attention mechanism takes two inputs current step -- decoder_hidden_state and all the encoder_outputs.
        '''
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, axis=1)

        # mask from encoder
        enc_mask = tf.expand_dims(enc_mask, axis=-1)

        # score shape: (batch_size, input_length, 1)
        if self.scoring_function == 'dot':
            # Implementing Dot score function
            score = self.dot([encoder_output, decoder_hidden_state])
        elif self.scoring_function == 'general':
            # Implementing General score function here
            score = tf.keras.layers.Dot(axes=[2, 2])([self.wa(encoder_output), decoder_hidden_state])
        elif self.scoring_function == 'concat':
            # Implementing General score function here
            decoder_output = tf.tile(decoder_hidden_state, [1, encoder_output.shape[1], 1])
            score = self.va(self.wa(tf.concat((decoder_output, encoder_output), axis=-1)))

        score = score + (tf.cast(tf.math.equal(enc_mask, False), score.dtype) * -1e9)
        # shape: (batch_size, input_length, 1)
        attention_weights = Softmax(axis=1)(score)

        enc_mask = tf.cast(enc_mask, attention_weights.dtype)
        # masking attention weights
        attention_weights = attention_weights * enc_mask
        context_vector = attention_weights * encoder_output

        # shape = (batch_size, dec lstm units)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

input_length=10
batch_size=16
att_units=32
scoring_fun = 'concat'

state_h=tf.random.uniform(shape=[batch_size,att_units])
encoder_output=tf.random.uniform(shape=[batch_size,input_length,att_units])
encoder_mask = tf.cast(tf.random.uniform(shape=[batch_size, input_length]), dtype=bool)
#encoder_mask = tf.cast(tf.zeros(shape=[batch_size, input_length]), dtype=bool)
attention=Attention(scoring_fun,att_units)
context_vector,attention_weights=attention(state_h,encoder_output, encoder_mask)


class OneStepDecoder(tf.keras.Model):
    def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
        """
        In this layer calculate the ooutput for a single timestep
        """
        super().__init__()

        # Initialize decoder embedding layer, LSTM and any other objects needed
        self.vocab_size = tar_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.embedding = Embedding(self.vocab_size, self.embedding_dim,
                                   trainable=True, mask_zero=True,
                                   name="Att_Dec_Embedding")

        # self.embedding = Embedding(self.vocab_size, self.embedding_dim,
        #                             embeddings_initializer=tf.keras.initializers.Constant(ans_embedding_matrix), trainable=False,
        #                             input_length=self.input_length, mask_zero=True, name="Att_Dec_Embedding")

        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, name="Att_Dec_LSTM")
        self.fc = Dense(self.vocab_size)
        # attention layers
        self.attention = Attention(self.score_fun, self.att_units)


    def call(self,input_to_decoder, encoder_output, state_h, state_c, enc_mask):
        '''
        Calling this function by passing decoder input for a single timestep, encoder output and encoder final states
        '''
        # shape: (batchsize, input_length, embedding dim)
        embedded_input = self.embedding(input_to_decoder)
        # shape: (batch_size, dec lstm units)
        context_vector, attention_weights = self.attention(state_h, encoder_output, enc_mask)
        # (batch_size, 1, dec lstm units)
        decoder_input = tf.concat([tf.expand_dims(context_vector, 1), embedded_input], axis=-1)
        # output shape: (batch size, input length, lstm units), state shape: (batch size, lstm units)
        decoder_output, dec_state_h, dec_state_c = self.lstm(decoder_input, initial_state=[state_h, state_c])
        # (batch_size, lstm units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        # (batch size, vocab size)
                # self.fc = Dense(self.vocab_size)

        output = self.fc(decoder_output)

        return output, dec_state_h, dec_state_c, attention_weights, context_vector

tar_vocab_size = 13
embedding_dim = 12
input_length = 10
dec_units = 16
att_units = 16
batch_size = 16
score_fun = 'concat'

onestepdecoder = OneStepDecoder(tar_vocab_size, embedding_dim, input_length, dec_units, score_fun, att_units)
input_to_decoder = tf.random.uniform(shape=(batch_size, 1), maxval=10, minval=0, dtype=tf.int32)
# input_to_decoder = tf.zeros(shape=(batch_size,1))
encoder_output = tf.random.uniform(shape=[batch_size, input_length, dec_units])
state_h = tf.random.uniform(shape=[batch_size, dec_units])
state_c = tf.random.uniform(shape=[batch_size, dec_units])
encoder_mask = tf.cast(tf.random.uniform(shape=[batch_size, input_length]), dtype=bool)
output, state_h, state_c, attention_weights, context_vector = onestepdecoder(input_to_decoder, encoder_output, state_h,
                                                                           state_c, encoder_mask)
# print(output.shape)

class Decoder(tf.keras.Model):
    def __init__(self, out_vocab_size, embedding_dim, input_length, dec_units, score_fun, att_units):
        # Intialize necessary variables and create an object from the class onestepdecoder
        super().__init__()
        self.vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units

        # Initializing onestepdecoder layer
        self.onestepdecoder = OneStepDecoder(self.vocab_size, self.embedding_dim, self.input_length,
                                             self.dec_units, self.score_fun, self.att_units)

    @tf.function
    def call(self, input_to_decoder, encoder_output, decoder_hidden_state, decoder_cell_state, enc_mask):
        # Initializing an empty Tensor array, that will store the outputs at each and every time step
        all_outputs = tf.TensorArray(tf.float32, size=input_to_decoder.shape[1], name="Output_array")

        # Iterate till the length of the decoder input
        for timestep in range(input_to_decoder.shape[1]):
            # Calling onestepdecoder for each token in decoder_input
            output, decoder_hidden_state, decoder_cell_state, attention_weights, context_vector = self.onestepdecoder(
                input_to_decoder[:, timestep:timestep + 1], encoder_output, decoder_hidden_state, decoder_cell_state,
                enc_mask)
            output_one_hot = tf.one_hot(tf.argmax(output, axis=-1), depth=self.vocab_size)

            # Store the output in tensorarray
            all_outputs = all_outputs.write(timestep, output)

        all_outputs = tf.transpose(all_outputs.stack(), [1, 0, 2])

        # Return the tensor array
        return all_outputs


out_vocab_size=13
embedding_dim=12
input_length=11
dec_units=16
att_units=16
batch_size=16
score_fun = 'concat'

target_sentences=tf.random.uniform(shape=(batch_size,input_length),maxval=10,minval=0,dtype=tf.int32)
#target_sentences=tf.zeros(shape=(batch_size,input_length),dtype=tf.int32)
encoder_output=tf.random.uniform(shape=[batch_size,input_length,dec_units])
state_h=tf.random.uniform(shape=[batch_size,dec_units])
state_c=tf.random.uniform(shape=[batch_size,dec_units])
encoder_mask = tf.cast(tf.random.uniform(shape=[batch_size, input_length]), dtype=bool)
decoder=Decoder(out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units)
output=decoder(target_sentences,encoder_output, state_h, state_c, encoder_mask)
print(output.shape)


class Encoder_decoder(tf.keras.Model):

    def __init__(self, **params):
        super().__init__()
        self.inp_vocab_size = params['inp_vocab_size']
        self.out_vocab_size = params['out_vocab_size']
        self.embedding_size = params['embedding_size']
        self.lstm_size = params['lstm_units']
        self.input_length = params['input_length']
        self.batch_size = params['batch_size']
        self.score_fun = params["score_fun"]

        # Create encoder object
        self.encoder = Encoder(self.inp_vocab_size + 1, embedding_size=self.embedding_size, lstm_size=self.lstm_size,
                               input_length=self.input_length)

        # Create decoder object
        self.decoder = Decoder(self.out_vocab_size + 1, embedding_dim=self.embedding_size,
                               input_length=self.input_length,
                               dec_units=2 * self.lstm_size, score_fun=self.score_fun, att_units=2 * self.lstm_size)
        self.output_layer = tf.keras.layers.Dense(self.out_vocab_size + 1, activation=None)

    def call(self, data):
        '''
        Calling the model with ([encoder input, decoder input], decoder outpur)
        '''
        input, output = data[0], data[1]

        enc_initial_states = self.encoder.initialize_states(self.batch_size)
        enc_out, enc_state_h, enc_state_c, enc_mask = self.encoder(input, enc_initial_states)

        dec_out = self.decoder(output, enc_out, enc_state_h, enc_state_c, enc_mask)
        output_layer = tf.keras.layers.Dense(ANS_VOCAB_SIZE, activation=None)  # No activation function
        logits = self.output_layer(dec_out)
        # Apply the final layer to the decoder output
        # decoder_output = output_layer(dec_out)
        # decoder_output = tf.keras.layers.Softmax()( decoder_output)
        return dec_out

# Earlystop callback
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=1)

# Checkpoint callback
checkpoint_filepath = "./checkpoint.weights.h5"  # Adjust filepath here
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', verbose=1, \
                                                         save_best_only=True, save_weights_only=True)

# Tensorboard
import datetime
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def custom_lossfunction(real, pred_logits):
    # Ensure real is of integer type
    real = tf.cast(real, dtype=tf.int32)

    # Apply softmax to logits
    pred_probs = tf.nn.softmax(pred_logits)
    # print(real)
    # print(pred_probs)

    # Compute cross-entropy loss
    loss = tf.keras.losses.sparse_categorical_crossentropy(real, pred_probs, from_logits=False)

    # Masking to handle padded sequences
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), dtype=loss.dtype)
    loss *= mask

    # Compute mean loss over non-padded tokens
    mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

    return mean_loss

print(encoder_input_data.shape)

print(decoder_input_data.shape)
print(decoder_output_data.shape)
model = Encoder_decoder(inp_vocab_size=QSN_VOCAB_SIZE,
                        out_vocab_size=ANS_VOCAB_SIZE,
                        embedding_size=EMBEDDING_SIZE,
                        lstm_units=LSTM_UNITS,
                        input_length=MAXLEN,
                        batch_size=BATCH_SIZE,
                        score_fun="concat")
adam_optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=adam_optimizer, loss=custom_lossfunction, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit([encoder_input_data , decoder_input_data], decoder_output_data,
          epochs=800, batch_size=4, verbose=1,
          )

# model.compile(optimizer=keras.optimizers.RMSprop(), loss=custom_lossfunction)
#
# model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=4, epochs=600 )

model.summary()

def predict(input_sentence, model):
    '''
    For a given question this function return a answer
    '''
    # Preparing input data
    input_tokens = enc_tokenizer.texts_to_sequences(input_sentence)
    input_sequence = pad_sequences(input_tokens, maxlen=MAXLEN, padding='post')

    batch_size = input_sequence.shape[0]

    # Getting encoder output and states
    enc_initial_states = model.encoder.initialize_states(batch_size)
    enc_out, enc_state_h, enc_state_c, enc_mask = model.encoder(input_sequence, enc_initial_states)
    state_h, state_c = enc_state_h, enc_state_c

    # Sending '<start>' as 1st word of decoder
    target_word = np.zeros((batch_size, 1))
    target_word[:, 0] = dec_tokenizer.word_index['<start>']

    end_token = dec_tokenizer.word_index["<end>"]

    outwords = []
    while np.array(outwords).shape[0] < MAXLEN:
        # decoder layer, intial states are encoder's final states
        output, dec_state_h, dec_state_c, attention_weights, _ = model.decoder.onestepdecoder(target_word, enc_out,
                                                                                              state_h, state_c,
                                                                                              enc_mask)

        out_word_index = np.argmax(output, -1)
        outwords.append(out_word_index)

        # current output word is input word for next timestamp
        target_word = np.zeros((batch_size, 1))
        target_word[:, 0] = out_word_index

        # current out states are input states for next timestamp
        state_h, state_c = dec_state_h, dec_state_c
        if (np.array(outwords[-1]) == end_token).all():
            break

    sentences = []
    outwords = np.array(outwords)
    print(outwords)

    # Convert the Tokenizer object into a numpy array
    sequences = np.array(list(dec_tokenizer.word_index.items()))

    for sent_token in outwords.T:
        current_sent = ""
        for ind in sent_token:
            if ind != end_token:
                current_sent += dec_tokenizer.index_word[ind] + " "
            else:
                current_sent += "<end>"
                break
        sentences.append(current_sent.strip())

    return sentences


print(predict(['Tell me about yourself'], model))

model.compile(optimizer=keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()
model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=4, epochs=100 )
model.save( 'model.h5' )