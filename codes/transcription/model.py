import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class EncoderDecoderModel(Model):
    def __init__(self, vocab_size, encoder_units=256, decoder_units=256, n_fft=1024):
        """
        Initializes the encoder-decoder model with LSTM layers for processing STFT and waveform inputs.
        
        Parameters:
        - vocab_size (int): Size of the vocabulary for the output.
        - encoder_units (int): Number of units for the LSTM in the encoder.
        - decoder_units (int): Number of units for the LSTM in the decoder.
        - n_fft (int): FFT window size for the waveform input shape.
        """
        super(EncoderDecoderModel, self).__init__()
        
        # Encoder for STFT input
        self.stft_input = Input(shape=(None, 1 + n_fft // 2, 1), name='stft_input')
        self.stft_lstm_input = layers.TimeDistributed(layers.Flatten())
        self.stft_lstm = layers.LSTM(encoder_units, return_state=True, return_sequences=False)
        
        # Encoder for waveform input
        self.waveform_input = Input(shape=(None, n_fft, 1), name='waveform_input')
        self.waveform_lstm_input = layers.TimeDistributed(layers.Flatten())
        self.waveform_lstm = layers.LSTM(encoder_units, return_state=True, return_sequences=False)
        
        # Concatenate layers for encoder states
        self.concat_state_h = layers.Concatenate()
        self.concat_state_c = layers.Concatenate()
        
        # Decoder embedding and LSTM
        self.decoder_input = Input(shape=(None,), name='decoder_input')
        self.decoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=encoder_units)
        self.decoder_lstm = layers.LSTM(decoder_units, return_sequences=True, return_state=True)
        
        # Dense output layer
        self.decoder_dense = layers.Dense(vocab_size, activation='softmax')
    
    def call(self, inputs):
        """
        Forward pass of the model.
        
        Parameters:
        - inputs (list): A list containing [stft_input, waveform_input, decoder_input].
        
        Returns:
        - decoder_output: The output of the decoder with token probabilities.
        """
        stft_input, waveform_input, decoder_input = inputs
        
        # Process STFT input
        stft_x = self.stft_lstm_input(stft_input)
        _, stft_state_h, stft_state_c = self.stft_lstm(stft_x)
        
        # Process waveform input
        waveform_x = self.waveform_lstm_input(waveform_input)
        _, waveform_state_h, waveform_state_c = self.waveform_lstm(waveform_x)
        
        # Concatenate encoder states
        encoder_state_h = self.concat_state_h([stft_state_h, waveform_state_h])
        encoder_state_c = self.concat_state_c([stft_state_c, waveform_state_c])
        
        # Embed decoder input
        embedded_decoder_input = self.decoder_embedding(decoder_input)
        
        # Decoder forward pass
        decoder_output, _, _ = self.decoder_lstm(embedded_decoder_input, initial_state=[encoder_state_h, encoder_state_c])
        decoder_output = self.decoder_dense(decoder_output)
        
        return decoder_output