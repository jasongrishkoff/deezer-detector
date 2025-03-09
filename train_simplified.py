"""
Simplified training script using our modified audio loader
"""

import os
import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ensure the current directory is in the path
sys.path.insert(0, os.getcwd())

from loader.global_variables import *
from loader.audio_modified import SimpleAudioLoader
from model.simple_cnn import SimpleSpectrogramCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="gpu to evaluate on", type=int, default=0)
    parser.add_argument("--config", help="config file", type=str, default="specnn_amplitude")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=30)
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
    
    # Basic config
    config = {
        'batch_size': args.batch_size,
        'audio_slice': 3.0,  # seconds
        'sr': 44100,
        'repeat': 5,
        
        # CNN parameters
        'hidden': [16, 32, 64, 128, 256, 512],
        'strides': [2, 2, 2, 2, 2, 2],
        'kernel': [3, 3, 3, 3, 3, 3],
        'use_maxpool': True,
        'head_dense': 128,
        'conv_activation': 'relu',
        'dense_activation': 'swish',
        'use_batch_norm': True,
        'global_pooling': 'average',
        
        # Preprocessing
        'effects': ['mono', 'affine', 'stft_db', 'normalise', 'slice_hf'],
        'normalise_mean': -4.0,
        'normalise_std': 3.0,
        'hf_cut': 16000,
    }
    
    # Create data loader
    loader = SimpleAudioLoader(POS_DB_PATH, NEG_DB_PATH, config, split_path=SPLIT_PATH)
    
    # Create datasets
    train_ds = loader.prepare_tensorflow_dataset('train')
    val_ds = loader.prepare_tensorflow_dataset('validation')
    
    # Create model
    # This is a placeholder - we need to actually implement the appropriate preprocessing
    # and create the correct input shape for the model
    input_shape = (config['audio_slice'] * config['sr'], 2)  # Stereo audio
    model = SimpleSpectrogramCNN(input_shape, config)
    
    # Compile model
    model.m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()]
    )
    
    # Create callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f"{WEIGHTS_PATH}/{args.config}",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            factor=0.1,
            min_lr=1e-6
        )
    ]
    
    # Train model
    model.m.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.m.save(f"{WEIGHTS_PATH}/{args.config}_final")
    print(f"Model saved to {WEIGHTS_PATH}/{args.config}_final")

if __name__ == "__main__":
    main()
