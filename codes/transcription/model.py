import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import gc

class EncoderDecoderModel(Model):
    def __init__(self, vocab_size, encoder_units=256, decoder_units=256, n_fft=1024, max_seq_length=200):
        super(EncoderDecoderModel, self).__init__()
        self.stft_input = Input(shape=(None, 1 + n_fft // 2, 1), name='stft_input')
        self.stft_lstm_input = layers.TimeDistributed(layers.Flatten())
        self.stft_lstm = layers.LSTM(encoder_units, return_state=True, return_sequences=False)
        
        self.waveform_input = Input(shape=(None, n_fft, 1), name='waveform_input')
        self.waveform_lstm_input = layers.TimeDistributed(layers.Flatten())
        self.waveform_lstm = layers.LSTM(encoder_units, return_state=True, return_sequences=False)
        
        self.concat_state_h = layers.Concatenate()
        self.concat_state_c = layers.Concatenate()
        
        self.state_reducer_h = layers.Dense(256)
        self.state_reducer_c = layers.Dense(256)
        
        self.decoder_input = Input(shape=(max_seq_length,), name='decoder_input')
        self.decoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=encoder_units)
        self.decoder_lstm = layers.LSTM(decoder_units, return_sequences=True, return_state=True)
        
        self.decoder_dense = layers.Dense(vocab_size, activation='softmax')
    
    def call(self, inputs, training=True):
        stft_input, waveform_input, decoder_input = inputs['stft_input'], inputs['waveform_input'], inputs['decoder_input']
        
        stft_x = self.stft_lstm_input(stft_input)
        _, stft_state_h, stft_state_c = self.stft_lstm(stft_x)
        
        waveform_x = self.waveform_lstm_input(waveform_input)
        _, waveform_state_h, waveform_state_c = self.waveform_lstm(waveform_x)
        
        encoder_state_h = self.concat_state_h([stft_state_h, waveform_state_h])
        encoder_state_c = self.concat_state_c([stft_state_c, waveform_state_c])
        
        encoder_state_h = self.state_reducer_h(encoder_state_h)
        encoder_state_c = self.state_reducer_c(encoder_state_c)
        
        decoder_hidden_state = [encoder_state_h, encoder_state_c]
        seq_length = tf.shape(decoder_input)[1]

        predictions = tf.TensorArray(dtype=tf.float32, size=seq_length)

        current_input = decoder_input[:, 0]  

        for t in tf.range(1, seq_length):
            embedded_input = self.decoder_embedding(tf.expand_dims(current_input, axis=1))
            decoder_output, state_h, state_c = self.decoder_lstm(embedded_input, initial_state=decoder_hidden_state)
            output_token = self.decoder_dense(decoder_output)

            predictions = predictions.write(t - 1, output_token)

            if training: 
                current_input = decoder_input[:, t]
            else:  
                current_input = tf.cast(tf.argmax(output_token, axis=-1)[:, 0], dtype=tf.int32)

            decoder_hidden_state = [state_h, state_c]

        predictions = predictions.stack()
        predictions = tf.squeeze(predictions, axis=-2)
        predictions = tf.transpose(predictions, [1, 0, 2])

        gc.collect()
        
        return predictions
        
