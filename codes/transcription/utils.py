import tensorflow as tf
import librosa
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import os
from .tokenizer import vocabulary
from .classes import TokenizerConfig
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
N_FFT = 1024  
HOP_LENGTH = N_FFT
BATCH_SIZE = 8

tokens = vocabulary(TokenizerConfig()).from_midi_tags()

tokenizer = Tokenizer(filters='', lower=False)  
tokenizer.fit_on_texts(tokens)

def load_audio(file_path, n_fft=N_FFT, hop_length=HOP_LENGTH):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude = np.abs(stft_result)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    
    stft_db = np.expand_dims(stft_db.T, axis=-1)  # Shape: (num_frames, frequency_bins, 1)
    
    num_frames = stft_db.shape[0]
    segment_length = n_fft
    
    waveform_segments = []
    for i in range(0, num_frames * hop_length, hop_length):
        segment = audio[i:i + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
        waveform_segments.append(segment)
    
    waveform_segments = np.array(waveform_segments)
    waveform_segments = np.expand_dims(waveform_segments, axis=-1)  # Shape: (num_frames, n_fft, 1)
    
    return stft_db, waveform_segments

def tokenize_midi(midi_string):
    with open(midi_string,'r') as f:
        midi_string = f.read()
    
    tokenized = tokenizer.texts_to_sequences([midi_string.split()])[0]
    return np.array(tokenized)

def generator(file_paths, midi_paths):
    for file_path, midi_string in zip(file_paths, midi_paths):
        stft_db, waveform_segments = load_audio(file_path)
        tokenized_midi = tokenize_midi(midi_string)
        
        decoder_input = tokenized_midi[:-1] 
        target_output = tokenized_midi[1:]  
        yield ({'stft_input': stft_db, 'waveform_input': waveform_segments, 'decoder_input': decoder_input}, target_output)

class datasets:
    def __init__(self,root_file, root_midi, batch_size=BATCH_SIZE):
        self.file_paths, self.midi_paths = [], []
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        for root, _, files in os.walk(root_file):
            for file in files:
                self.file_paths.append(os.path.join(root,file))
        for root, _, files in os.walk(root_midi):
            for file in files:
                self.midi_paths.append(os.path.join(root,file))
                
        div = round(len(self.file_paths)*0.8)
        
        self.train = self.data_generator(
            self.file_paths[:div],self.midi_paths[:div]
        )
        self.valid = self.data_generator(
            self.file_paths[div:((len(self.file_paths))-div)//2+div],self.midi_paths[div:((len(self.file_paths))-div)//2+div]
        )
        self.test = self.data_generator(
            self.file_paths[((len(self.file_paths))-div)//2+div:],self.midi_paths[((len(self.file_paths))-div)//2+div:]
        )
             
    def data_generator(self,file_paths,midi_paths):
        dataset = tf.data.Dataset.from_generator(
            generator,
            args=(file_paths,midi_paths),
            output_signature=(
                {
                    'stft_input': tf.TensorSpec(shape=(None, 1 + N_FFT // 2,1), dtype=tf.float32),
                    'waveform_input': tf.TensorSpec(shape=(None, N_FFT,1), dtype=tf.float32),
                    'decoder_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                },
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=(
            {
            'stft_input': [None, 1 + N_FFT // 2,1],
            'waveform_input': [None, N_FFT,1],
            'decoder_input': [None],
        },
        [None]
    ))
        return dataset    
    
    def retrieve(self):
        return self.train, self.valid, self.test
    
def plot_training(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()