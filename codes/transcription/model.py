import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from .utils import from_midi_tags
import gc

class EncoderDecoderModel(Model):
    def __init__(self,encoder_units=256, decoder_units=256, n_fft=1024, max_seq_length=300):
        super(EncoderDecoderModel, self).__init__()
        tokens = from_midi_tags()
        self.tokenizer = Tokenizer(filters='', lower=False)  
        self.tokenizer.fit_on_texts(tokens)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.max_seq_length = max_seq_length
        self.n_fft = n_fft
        self.eos_token_id = self.tokenizer.word_index['EOS']
        
        self.stft_input = Input(shape=(None, 1 + n_fft // 2, 1), name='stft_input')
        self.stft_conv_1 = layers.Conv2D(filters=256,kernel_size=(3, 3), activation='relu')
        self.stft_pool_1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.stft_conv_2 = layers.Conv2D(filters=256,kernel_size=(3, 3), activation='relu')
        self.stft_pool_2 = layers.MaxPooling2D(pool_size=(2, 2))
        
        self.stft_lstm_input = layers.TimeDistributed(layers.Flatten())
        self.stft_lstm = layers.LSTM(encoder_units, return_state=True, return_sequences=False)
        
        self.waveform_input = Input(shape=(None, n_fft), name='waveform_input')
        self.waveform_conv_1 = layers.Conv1D(filters=256,kernel_size=3, activation='relu')
        self.waveform_pool_1 = layers.MaxPooling1D(pool_size=2)
        self.waveform_conv_2 = layers.Conv1D(filters=256,kernel_size=3, activation='relu')
        self.waveform_pool_2 = layers.MaxPooling1D(pool_size=2)
        
        self.waveform_lstm_input = layers.TimeDistributed(layers.Flatten())
        self.waveform_lstm = layers.LSTM(encoder_units, return_state=True, return_sequences=False)
        
        self.concat_state_h = layers.Concatenate(name='state_h')
        self.concat_state_c = layers.Concatenate(name='state_c')
        
        self.state_reducer_h = layers.Dense(encoder_units)
        self.state_reducer_c = layers.Dense(encoder_units)
        
        self.decoder_input = Input(shape=(max_seq_length,), name='decoder_input')
        self.decoder_embedding = layers.Embedding(input_dim=self.vocab_size, output_dim=decoder_units)
        self.decoder_lstm = layers.LSTM(decoder_units, return_sequences=True, return_state=True)
        
        self.decoder_dense = layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, training=True):
        stft_input, waveform_input, decoder_input = inputs['stft_input'], inputs['waveform_input'], inputs['decoder_input']
        
        stft_x = self.stft_conv_1(stft_input)
        stft_x = self.stft_pool_1(stft_x)
        stft_x = self.stft_conv_2(stft_x)
        stft_x = self.stft_pool_2(stft_x)
        stft_x = self.stft_lstm_input(stft_x)
        _, stft_state_h, stft_state_c = self.stft_lstm(stft_x)
        gc.collect()
        
        waveform_x = self.waveform_conv_1(waveform_input)
        waveform_x = self.waveform_pool_1(waveform_x)
        waveform_x = self.waveform_conv_2(waveform_x)
        waveform_x = self.waveform_pool_2(waveform_x)
        waveform_x = self.waveform_lstm_input(waveform_x)
        _, waveform_state_h, waveform_state_c = self.waveform_lstm(waveform_x)
        gc.collect()
        
        encoder_state_h = self.concat_state_h([stft_state_h, waveform_state_h])
        encoder_state_c = self.concat_state_c([stft_state_c, waveform_state_c])
        
        encoder_state_h = self.state_reducer_h(encoder_state_h)
        encoder_state_c = self.state_reducer_c(encoder_state_c)
        
        initial_state = [encoder_state_h, encoder_state_c]
        seq_length = self.max_seq_length

        predictions = tf.TensorArray(dtype=tf.float32, size=seq_length)

        current_input = decoder_input[:, 0]  

        for t in range(1, seq_length):
            embedded_input = self.decoder_embedding(tf.expand_dims(current_input, axis=1))
            decoder_output, state_h, state_c = self.decoder_lstm(embedded_input, initial_state=initial_state)
            output_token = self.decoder_dense(decoder_output)

            predictions = predictions.write(t - 1, output_token)

            if training: 
                current_input = decoder_input[:, t]
            else:  
                current_input = tf.argmax(output_token, axis=-1)[:, 0]

            initial_state = [state_h, state_c]
            gc.collect()

        predictions = predictions.stack()
        predictions = tf.squeeze(predictions, axis=-2)
        predictions = tf.transpose(predictions, [1, 0, 2])

        gc.collect()
        return predictions
    
    def build_graph(self):
        stft = Input(shape=(None, 1 + self.n_fft // 2,1), name='stft_input')
        waveform = Input(shape=(None, self.n_fft), name='waveform_input')
        decoder = Input(shape=(300,), name='decoder_input')
        
        dummy = {
            'stft_input': stft,
            'waveform_input': waveform,
            'decoder_input': decoder
        }
        return tf.keras.Model(inputs=[stft,waveform,decoder],
                              outputs=self(dummy))