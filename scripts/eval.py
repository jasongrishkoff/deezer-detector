"""
Modified evaluation script for AI-music detection
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

HOME = "/workspace"
POS_DB_PATH = HOME+"/human_examples"                  # Your original audio files
NEG_DB_PATH = HOME+"/deezer-detector/fma_rebuilt"      # Temporary path for restructured files
CODEC_DB_PATH = HOME+"/deezer-detector/fma_codec/"    # For codec-altered versions
WEIGHTS_PATH = HOME+"/deezer-detector/weights"        # Model checkpoints
RESULT_PATH = HOME+"/deezer-detector/results"         # Evaluation results
SPLIT_PATH = HOME+"/deezer-detector/data/dataset_split.npy"  # Train/val/test split
BR_PATH = HOME+"/deezer-detector/data/bitrates.npy"   # Bitrate information
CONF_PATH = HOME+"/deezer-detector/conf"              # Config files

from model.simple_cnn import SimpleCNN, SimpleSpectrogramCNN

# Reuse the CustomAudioLoader from train.py
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
   
    def create_dataset(self, split, encoder=None, batch_size=32, shuffle=True, augmenter=None):
        """Create a TensorFlow dataset for the given split"""
        print(f"Creating dataset for {split} split")
        
        if split not in self.split_paths:
            raise ValueError(f"Split '{split}' not found in data")
        
        paths = self.split_paths[split]
        
        # Filter paths based on encoder if specified
        if encoder is not None and encoder != "real":
            if encoder not in self.encoders:
                raise ValueError(f"Encoder '{encoder}' not found")
            print(f"Filtering dataset for encoder: {encoder}")
            neg_paths = paths['neg'][encoder]
            print(f"Found {len(paths['pos'])} positive and {len(neg_paths)} negative files for encoder {encoder}")
        else:
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

        # Create dataset based on whether we're filtering for a specific encoder or not
        if encoder == "real":
            # For real audio only - must match model output format
            paths_to_use = paths['pos']
            dataset = tf.data.Dataset.from_tensor_slices(paths_to_use)
            dataset = dataset.map(lambda x: load_audio(x, 1.0), num_parallel_calls=tf.data.AUTOTUNE)
            # Add encoder output format to match model expectation
            dataset = dataset.map(lambda x, y: (x, (y, tf.zeros(self.n_encoders))), num_parallel_calls=tf.data.AUTOTUNE)
        elif encoder is not None:
            # For a specific encoder - must match model output format
            paths_to_use = paths['neg'][encoder]
            dataset = tf.data.Dataset.from_tensor_slices(paths_to_use)
            dataset = dataset.map(lambda x: load_audio(x, 0.0), num_parallel_calls=tf.data.AUTOTUNE)
            # Create one-hot encoding for this specific encoder
            encoder_idx = self.encoders.index(encoder)
            dataset = dataset.map(
                lambda x, y: (x, (y, tf.one_hot(encoder_idx, depth=self.n_encoders))), 
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            # For mixed evaluation with all encoders
            # Create TF datasets for positive and negative samples
            positive_ds = tf.data.Dataset.from_tensor_slices(paths['pos'])
            positive_ds = positive_ds.map(lambda x: load_audio(x, 1.0), num_parallel_calls=tf.data.AUTOTUNE)
            
            # Combine negative datasets from all encoders
            neg_paths = []
            encoders_idx = []
            for idx, enc in enumerate(self.encoders):
                neg_paths.extend(paths['neg'][enc])
                encoders_idx.extend([idx] * len(paths['neg'][enc]))
                
            # Convert encoders_idx to tf.int32 to avoid type issues in one_hot encoding
            encoders_idx = [tf.constant(idx, dtype=tf.int32) for idx in encoders_idx]
            
            negative_ds = tf.data.Dataset.from_tensor_slices((neg_paths, encoders_idx))
            negative_ds = negative_ds.map(
                lambda x, y: (load_audio(x, 0.0)[0], (0.0, tf.one_hot(y, depth=self.n_encoders))), 
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Convert positive labels to proper format for multi-task learning
            positive_ds = positive_ds.map(
                lambda x, y: (x, (y, tf.zeros(self.n_encoders))),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Combine datasets
            dataset = tf.data.Dataset.concatenate(positive_ds, negative_ds)
            
        # Common dataset preparation
        if shuffle:
            buffer_size = 1000 if encoder is None else len(paths_to_use)
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        
        # Repeat samples for efficiency
        dataset = dataset.flat_map(
            lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(self.config.get('repeat', 5))
        )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file", type=str, default="specnn_amplitude")
    parser.add_argument("--weights", help="weights file, else defaults to config", type=str)
    parser.add_argument("--encoder", help="evaluate only for specific encoder", type=str, default="")
    parser.add_argument("--gpu", help="gpu to evaluate on", type=int, default=0)
    parser.add_argument("--steps", help="Number of evaluation steps", type=int, default=50)
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    args = parser.parse_args()

    # Use weights from config if not specified
    if not args.weights:
        args.weights = args.config
    
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
    global_conf = {
        'batch_size': args.batch_size,
        'audio_slice': 0.78,
        'repeat': 5,
        'sr': 44100,
        'fft': {
            'win': 2048,
            'hop': 512,
            'n_fft': 2048
        },
        'hidden': [16, 32, 64, 128, 256, 512],
        'strides': [2, 2, 2, 2, 2, 2],
        'kernel': [3, 3, 3, 3, 3, 3],
        'use_maxpool': True,
        'head_dense': 128,
        'conv_activation': 'relu',
        'dense_activation': 'swish',
        'use_batch_norm': True,
        'global_pooling': 'average',
        'effects': ['mono', 'affine', 'stft_db', 'normalise', 'slice_hf'],
        'normalise_mean': -4.0,
        'normalise_std': 3.0,
        'hf_cut': 16000,
        'name': args.config
    }
    
    # Create the data loader
    loader = CustomAudioLoader(POS_DB_PATH, NEG_DB_PATH, global_conf, SPLIT_PATH)
    
    # Create test dataset
    test_ds = loader.create_dataset('test', batch_size=global_conf['batch_size'])
    
    # Get input shape from a sample batch
    for x_batch, y_batch in test_ds.take(1):
        input_shape = x_batch.shape[1:]
        break
    
    print(f"Input shape: {input_shape}")
    
    # Create model based on configuration
    n_encoders = loader.n_encoders  # Use exact number of encoders
    
    if 'use_raw' in global_conf:
        model = SimpleCNN(input_shape, global_conf, detect_encoder=n_encoders)
    else:
        model = SimpleSpectrogramCNN(input_shape, global_conf, detect_encoder=n_encoders)
    
    # Load weights - handle directory vs file path
    weights_path = os.path.join(WEIGHTS_PATH, args.weights)
    if os.path.isdir(weights_path):
        weights_path = os.path.join(weights_path, "variables", "variables")
        print(f"Loading weights from directory: {weights_path}")
    else:
        print(f"Loading weights from file: {weights_path}")
    
    try:
        model.m.load_weights(weights_path)
    except:
        # Try loading the final model if available
        weights_path_final = os.path.join(WEIGHTS_PATH, f"{args.weights}_final")
        if os.path.exists(weights_path_final):
            print(f"Trying alternate weights path: {weights_path_final}")
            model.m.load_weights(weights_path_final)
    
    # Compile model
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
    
    # Evaluate on full test set
    print("\nEvaluating on full test set:")
    results = {}
    results['all'] = model.m.evaluate(test_ds, steps=args.steps)
    
    # Evaluate on each encoder separately
    for encoder in loader.encoders:
        print(f"\nEvaluating on encoder: {encoder}")
        encoder_ds = loader.create_dataset('test', encoder=encoder, batch_size=global_conf['batch_size'])
        results[encoder] = model.m.evaluate(encoder_ds, steps=args.steps)
    
    # Evaluate on real data only
    print("\nEvaluating on real data only:")
    real_ds = loader.create_dataset('test', encoder="real", batch_size=global_conf['batch_size'])
    results['real'] = model.m.evaluate(real_ds, steps=args.steps)
    
    # Save results
    os.makedirs(RESULT_PATH, exist_ok=True)
    np.save(os.path.join(RESULT_PATH, f"{args.config}_results.npy"), results)
    print(f"\nResults saved to {os.path.join(RESULT_PATH, args.config+'_results.npy')}")
    
    # Print summary
    print("\nEvaluation results summary:")
    print(f"Model: {args.config}")
    for key, value in results.items():
        if key == 'all':
            print(f"Overall accuracy: {value[1]:.4f}")
        elif key == 'real':
            print(f"Real audio accuracy: {value[1]:.4f}")
        else:
            print(f"Encoder {key} accuracy: {value[1]:.4f}")

if __name__ == "__main__":
    main()
