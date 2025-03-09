"""
Modified training script for AI-music detection
Includes fixes for file path handling and dataset loading
"""

import os
import argparse
import sys
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.getcwd())

from loader.global_variables import *
from loader.config import ConfLoader
from model.simple_cnn import SimpleCNN, SimpleSpectrogramCNN

class CustomAudioLoader:
    """Modified audio loader that handles the actual file paths properly"""
    
    def __init__(self, pos_db_path, neg_db_path, config, split_path):
        self.config = config
        self.augmenter = None  # Initialize augmenter attribute
        
        # Load the split data
        self.split_data = np.load(split_path, allow_pickle=True).item()
        
        # Find all audio files in the positive directory
        self.pos_files = {}
        for pattern in ['*.mp3', '*.wav']:
            for file_path in glob(os.path.join(pos_db_path, pattern)):
                filename = os.path.basename(file_path)
                self.pos_files[filename] = file_path
            
            # Also search recursively
            for file_path in glob(os.path.join(pos_db_path, '**', pattern), recursive=True):
                filename = os.path.basename(file_path)
                self.pos_files[filename] = file_path
        
        print(f"Found {len(self.pos_files)} positive audio files")
        
        # Find all encoders in the negative directory
        self.encoders = []
        for item in os.listdir(neg_db_path):
            if os.path.isdir(os.path.join(neg_db_path, item)):
                self.encoders.append(item)
        
        self.n_encoders = len(self.encoders)
        print(f"Found {self.n_encoders} encoders: {self.encoders}")
        
        # Find all audio files for each encoder
        self.neg_files = {}
        for encoder in self.encoders:
            self.neg_files[encoder] = {}
            
            for pattern in ['*.mp3', '*.wav']:
                # Direct files
                for file_path in glob(os.path.join(neg_db_path, encoder, pattern)):
                    filename = os.path.basename(file_path)
                    self.neg_files[encoder][filename] = file_path
                
                # Recursive search
                for file_path in glob(os.path.join(neg_db_path, encoder, '**', pattern), recursive=True):
                    filename = os.path.basename(file_path)
                    self.neg_files[encoder][filename] = file_path
            
            print(f"Found {len(self.neg_files[encoder])} files for encoder {encoder}")
        
        # Build dataset split mappings
        self.split_paths = {}
        for split in self.split_data:
            self.split_paths[split] = self._map_split_to_paths(self.split_data[split])
    
    def _map_split_to_paths(self, filenames):
        """Map filenames in split to actual file paths"""
        result = {
            'pos': [],
            'neg': {}
        }
        
        for encoder in self.encoders:
            result['neg'][encoder] = []
        
        for fname in filenames:
            # Check positive files
            if fname in self.pos_files:
                result['pos'].append(self.pos_files[fname])
            
            # Check negative files for each encoder
            for encoder in self.encoders:
                if fname in self.neg_files[encoder]:
                    result['neg'][encoder].append(self.neg_files[encoder][fname])
        
        return result
   
    def create_dataset(self, split, batch_size=32, shuffle=True, augmenter=None):
        """Create a TensorFlow dataset for the given split"""
        print(f"Creating dataset for {split} split")
        
        if split not in self.split_paths:
            raise ValueError(f"Split '{split}' not found in data")
        
        paths = self.split_paths[split]
        print(f"Found {len(paths['pos'])} positive and {sum([len(paths['neg'][enc]) for enc in self.encoders])} negative files")
        
        # Set augmenter if provided
        if augmenter is not None:
            self.augmenter = augmenter
        
        def audio_to_spectrogram(audio, config):
            """Convert audio to spectrogram using the configuration settings"""
            # Handle mono conversion if needed
            if 'mono' in config['effects']:
                # Convert stereo to mono (average channels)
                audio = tf.reduce_mean(audio, axis=-1, keepdims=True)
            
            # Apply STFT
            if any(effect.startswith('stft') for effect in config['effects']):
                # Get FFT parameters
                n_fft = config['fft']['n_fft']
                hop_length = config['fft']['hop']
                win_length = config['fft']['win']
                
                # Calculate STFT - make sure audio is 1D for STFT
                audio_1d = tf.squeeze(audio)
                
                # Handle case where audio might be 2D (stereo)
                if len(audio_1d.shape) > 1:
                    # Average channels for STFT if still stereo
                    audio_1d = tf.reduce_mean(audio_1d, axis=-1)
                
                stft = tf.signal.stft(
                    audio_1d, 
                    frame_length=win_length,
                    frame_step=hop_length,
                    fft_length=n_fft
                )
                
                # Convert to desired representation
                if 'stft_db' in config['effects']:
                    # Convert to power spectrogram (squared magnitude)
                    spectrogram = tf.abs(stft) ** 2
                    # Convert to dB scale
                    spectrogram = tf.math.log(tf.maximum(spectrogram, 1e-10))
                    # Normalize if specified
                    if 'normalise' in config['effects']:
                        mean = config.get('normalise_mean', 0.0)
                        std = config.get('normalise_std', 1.0)
                        spectrogram = (spectrogram - mean) / std
                    
                    # Cut high frequencies if specified
                    if 'slice_hf' in config['effects'] and 'hf_cut' in config:
                        max_freq = int((config['hf_cut'] * 2 / config['sr']) * (n_fft // 2 + 1))
                        spectrogram = spectrogram[..., :max_freq, :]
                    
                    # Add channel dimension
                    spectrogram = tf.expand_dims(spectrogram, -1)
                    return spectrogram
            
            # If no transformation or unsupported transformation, reshape audio for CNN
            # This should not happen with the spectrogram model, but just as fallback
            return tf.expand_dims(audio, -1)  # Add channel dimension

        # Create dataset from file paths
        def load_audio(path, is_real):
            """Load and preprocess audio file"""
            try:
                # Check if file is MP3 or WAV
                is_mp3 = tf.strings.regex_full_match(tf.strings.lower(path), ".*\\.mp3$")
                
                # Use torchaudio to load MP3 files through a py_function
                def load_with_torchaudio(file_path):
                    import torchaudio
                    import numpy as np
                    audio, sr = torchaudio.load(file_path.numpy().decode('utf-8'))
                    return np.transpose(audio.numpy(), (1, 0))  # Convert to [samples, channels]
                
                # Load audio based on file type
                audio = tf.py_function(
                    load_with_torchaudio,
                    [path],
                    tf.float32
                )
                
                # Ensure proper shape info is preserved
                audio.set_shape([None, 2])  # [samples, channels]
                
                # Handle audio length BEFORE converting to spectrogram
                target_length = int(self.config['audio_slice'] * 44100)
                audio_length = tf.shape(audio)[0]

                if audio_length > target_length:
                    start = tf.random.uniform([], 0, audio_length - target_length, dtype=tf.int32)
                    audio = audio[start:start+target_length]
                else:
                    # Pad if too short
                    padding = [[0, target_length - audio_length], [0, 0]]
                    audio = tf.pad(audio, padding)

                # Convert to spectrogram AFTER handling audio length
                features = audio_to_spectrogram(audio, self.config)
                
                # Apply augmentation if provided
                if self.augmenter is not None:
                    features = self.augmenter(features)
                
                return features, is_real
            except Exception as e:
                print(f"Error loading file: {path}, Error: {e}")
                
                # Return zeros with appropriate shape as fallback
                if 'stft_db' in self.config['effects']:
                    # For spectrogram, create dummy spectrogram shaped tensor
                    n_fft = self.config['fft']['n_fft']
                    max_freq = n_fft // 2 + 1
                    if 'slice_hf' in self.config['effects'] and 'hf_cut' in self.config:
                        max_freq = int((self.config['hf_cut'] * 2 / self.config['sr']) * max_freq)
                    
                    # Calculate time frames based on target length and hop size
                    hop_length = self.config['fft']['hop']
                    time_frames = (target_length - self.config['fft']['win']) // hop_length + 1
                    
                    # Return zeros with appropriate spectrogram shape [freq, time, channels]
                    return tf.zeros([max_freq, time_frames, 1]), is_real
                else:
                    # For raw audio, return zeros with [samples, channels]
                    return tf.zeros([target_length, 1]), is_real

        # Create TF datasets for positive and negative samples
        positive_ds = tf.data.Dataset.from_tensor_slices(paths['pos'])
        positive_ds = positive_ds.map(lambda x: load_audio(x, 1.0), num_parallel_calls=tf.data.AUTOTUNE)
        
        # Combine negative datasets from all encoders
        neg_paths = []
        encoders_idx = []
        for idx, encoder in enumerate(self.encoders):
            neg_paths.extend(paths['neg'][encoder])
            encoders_idx.extend([idx] * len(paths['neg'][encoder]))
            
        # Convert encoders_idx to tf.int32 to avoid type issues in one_hot encoding
        encoders_idx = [tf.constant(idx, dtype=tf.int32) for idx in encoders_idx]
        
        negative_ds = tf.data.Dataset.from_tensor_slices((neg_paths, encoders_idx))
        # For the encoder classification task, we need one-hot encoded labels
        negative_ds = negative_ds.map(
            lambda x, y: (load_audio(x, 0.0)[0], (0.0, tf.one_hot(y, depth=self.n_encoders))), 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Convert positive labels to proper format
        # Use -1 for encoder index, but create a one-hot vector of zeros for encoder output
        positive_ds = positive_ds.map(
            lambda x, y: (x, (y, tf.zeros(self.n_encoders))),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Combine datasets and prepare for training
        dataset = tf.data.Dataset.concatenate(positive_ds, negative_ds)
        
        if shuffle:
            dataset = dataset.shuffle(len(paths['pos']) + len(neg_paths), reshuffle_each_iteration=True)
        
        # Repeat samples for efficiency
        dataset = dataset.flat_map(
            lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(self.config.get('repeat', 5))
        )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="gpu to evaluate on", type=int, default=0)
    parser.add_argument("--config", help="config file", type=str, default="specnn_amplitude")
    parser.add_argument("--encoder", help="train on only one encoder", type=str, default="")
    parser.add_argument("--weights", help="continue training or fine-tune", type=str, default="")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=100)
    args = parser.parse_args()

    # Configure GPU
    if args.gpu >= 0:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
                print(f"Using GPU {args.gpu}")
            except RuntimeError as e:
                print(e)
    
    # Load configuration
    configuration = ConfLoader(CONF_PATH)
    configuration.load_model(args.config)
    global_conf = configuration.conf
    
    # Override batch size if specified
    if args.batch_size:
        global_conf['batch_size'] = args.batch_size
    
    print("Loading and preparing dataset...")
    
    # Create the custom loader
    loader = CustomAudioLoader(POS_DB_PATH, NEG_DB_PATH, global_conf, SPLIT_PATH)
    
    # Create TF datasets
    train_ds = loader.create_dataset('train', batch_size=global_conf['batch_size'])
    val_ds = loader.create_dataset('validation', batch_size=global_conf['batch_size'])
    
    # Get input shape from a sample batch
    for x_batch, y_batch in train_ds.take(1):
        input_shape = x_batch.shape[1:]
        break
    
    print(f"Input shape: {input_shape}")
    
    # Create model based on configuration
    n_encoders = None if args.encoder else loader.n_encoders  # Don't add +1, use exact number of encoders
    
    # Debug model configuration
    print(f"Creating model with n_encoders={n_encoders}")
    
    if 'use_raw' in global_conf:
        model = SimpleCNN(input_shape, global_conf, detect_encoder=n_encoders)
    else:
        # Make sure input_shape is appropriate for a CNN (has at least 3 dimensions: height, width, channels)
        if len(input_shape) < 3:
            print(f"Warning: Invalid input shape for CNN: {input_shape}")
            print("The model expects spectrogram data with shape [freq_bins, time_frames, channels]")
            print("Check the audio_to_spectrogram function to ensure it returns the correct shape")
            return
        model = SimpleSpectrogramCNN(input_shape, global_conf, detect_encoder=n_encoders)
    
    model.m.summary()
    
    # Load weights if specified
    if args.weights:
        print(f"Loading weights from {args.weights}")
        model.m.load_weights(args.weights)
    
    # Compile model
    if n_encoders:
        model.m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                'deepfake': keras.losses.BinaryCrossentropy(),
                'encoder': keras.losses.CategoricalCrossentropy(from_logits=True),
            },
            loss_weights={'deepfake': 1.0, 'encoder': 0.2},
            metrics={
                'deepfake': keras.metrics.BinaryAccuracy(),
                'encoder': keras.metrics.CategoricalAccuracy(),
            },
        )
    else:
        model.m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={'deepfake': keras.losses.BinaryCrossentropy()},
            metrics={'deepfake': keras.metrics.BinaryAccuracy()},
        )
    
    # Create callbacks
    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    model_name = f"{global_conf['name']}{args.encoder}"
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f"{WEIGHTS_PATH}/{model_name}",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=10,
            factor=0.1,
            mode='auto',
            min_lr=1e-5
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"{RESULT_PATH}/logs/{model_name}",
            histogram_freq=1
        )
    ]
    
    # Train model
    print("Starting training...")
    model.m.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.m.save(f"{WEIGHTS_PATH}/{model_name}_final")
    print(f"Training complete. Model saved to {WEIGHTS_PATH}/{model_name}_final")

if __name__ == "__main__":
    main()
