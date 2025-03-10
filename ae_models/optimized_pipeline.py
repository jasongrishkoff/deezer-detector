"""
GPU-optimized Pipeline for parallel audio processing
"""

import os
from time import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soxr
import torch
import torchaudio
from tqdm import tqdm

default_conf = {
    "DEVICE": "cuda",
    "DB_PATH": "/",
    "OUT_DB": "/",
    "BR_PATH": "/",
    "DATA_SR": 44100,
    "SR": 44100,  # target sr
    "TO_MONO": False,
    "AUGMENT": False,
    "MIN_DURATION": 3,  # seconds
    "MAX_DURATION": 40,
    "VERBOSE": False,
    "BITRATE": 320,  # Default bitrate when no bitrate data is available
    "BATCH_SIZE": 64,  # Process files in batches
    "NUM_WORKERS": 8,  # Number of parallel workers for loading files
}


class OptimizedPipeline:
    def __init__(self, paths, conf={}):
        self.paths = paths
        self.conf = dict(default_conf)
        self.conf.update(conf)
        
        # Load bitrates if available, otherwise use a default
        if os.path.exists(self.conf["BR_PATH"]):
            self.bitrates = np.load(self.conf["BR_PATH"], allow_pickle=True).item()
        else:
            print(f"Bitrate file not found: {self.conf['BR_PATH']}")
            print(f"Using default bitrate: {self.conf['BITRATE']}kbps")
            self.bitrates = {}

        self.global_t = time()
        
    def clock(self, ops_name):
        if self.conf["VERBOSE"]:
            current_time = time()
            print("{}: {:.2f}s".format(ops_name, current_time - self.global_t))
            self.global_t = current_time

    def load_audio_file(self, fpath):
        """Load and preprocess a single audio file"""
        try:
            fname = os.path.basename(fpath)
            ffolder = os.path.basename(os.path.dirname(fpath))
            
            audio_raw, sr = torchaudio.load(fpath)
            if sr != self.conf["SR"]:
                if self.conf["VERBOSE"]:
                    print(f"Resampling {fpath}: {sr} -> {self.conf['SR']}")
                audio_raw_rs = soxr.resample(audio_raw.T, sr, self.conf["SR"]).T
                audio_raw = torch.Tensor(audio_raw_rs)

            audio_raw = torch.squeeze(audio_raw)
            if audio_raw.shape[-1] < self.conf["SR"] * self.conf["MIN_DURATION"]:
                if self.conf["VERBOSE"]:
                    print(f"Track {fpath} < min duration, skipping")
                return None, None, None
                
            if len(audio_raw.shape) == 1:  # mono
                audio_raw = torch.stack([audio_raw, audio_raw], 0)
                
            if audio_raw.shape[-1] > self.conf["SR"] * self.conf["MAX_DURATION"]:
                audio_raw = audio_raw[:, :self.conf["SR"] * self.conf["MAX_DURATION"]]

            # Get bitrate from dictionary or use default
            audio_br = self.bitrates.get(fname, self.conf["BITRATE"])
            if isinstance(audio_br, (int, float)):
                audio_br = min(int(audio_br), 320)
            else:
                audio_br = self.conf["BITRATE"]
                
            return audio_raw, fname, ffolder, audio_br
            
        except Exception as err:
            print(f"Track {fpath} failed: [{type(err)}] {err}")
            return None, None, None, None

    def process_batch(self, file_paths, models, model_names, multi_codec, outputs_todo):
        """Process a batch of audio files in parallel"""
        # Load all audio files in the batch (can be done in parallel)
        loaded_data = []
        
        with ThreadPoolExecutor(max_workers=self.conf["NUM_WORKERS"]) as executor:
            results = list(executor.map(self.load_audio_file, file_paths))
            
        for result in results:
            if result[0] is not None:  # Skip failures
                audio_raw, fname, ffolder, audio_br = result
                # Move to device after loading
                audio_raw = audio_raw.to(self.conf["DEVICE"])
                loaded_data.append((audio_raw, fname, ffolder, audio_br))
        
        if not loaded_data:
            return
            
        # Process files with each model (using batch processing when possible)
        for m, m_name in zip(models, model_names):
            for codec_idx, c in enumerate(multi_codec):
                if c not in outputs_todo[m_name]:
                    continue
                    
                # Process each file
                with torch.no_grad():
                    for audio_raw, fname, ffolder, audio_br in loaded_data:
                        if 'autoencode_multi' in dir(m) and len(multi_codec) > 1:
                            # For models that support batch multi-codec processing
                            audios_rebuilt = m.autoencode_multi(audio_raw, [c])
                            audio_rebuilt = audios_rebuilt[0].to("cpu")
                        else:
                            # Standard autoencoding
                            audio_rebuilt = m.autoencode(audio_raw).to("cpu")
                            
                        # Save the result
                        out_path = os.path.join(self.conf["OUT_DB"], m_name + str(c), ffolder)
                        os.makedirs(out_path, exist_ok=True)
                        out_file = os.path.join(out_path, fname)
                        
                        torchaudio.save(
                            out_file,
                            audio_rebuilt,
                            sample_rate=self.conf["SR"],
                            channels_first=True,
                            format="mp3"
                        )

    def run_loop(self, models, model_names, multi_codec=[], has_cpu_preprocess=False, max_files=None):
        """Run the pipeline with GPU-optimized batch processing"""
        assert len(models) == len(model_names)
        if len(multi_codec) > 0:
            using_multi = True
            if model_names and len(model_names) > 1:
                print("param `model_names` will be ignored because of provided `multi_codec`")
            if len(models) > 1:
                raise ValueError("Does not make sense to have several models and `multi_codec` on.")
        else:
            multi_codec = ['']
            using_multi = False

        # Limit the number of files if specified
        paths_to_process = self.paths
        if max_files is not None and max_files < len(self.paths):
            paths_to_process = self.paths[:max_files]
            print(f"Processing a subset of {max_files} files out of {len(self.paths)}")

        # Figure out which outputs we need to generate
        outputs_todo = {m_name: [] for m_name in model_names}
        
        for fpath in tqdm(paths_to_process, desc="Checking existing outputs"):
            fname = os.path.basename(fpath)
            ffolder = os.path.basename(os.path.dirname(fpath))
            
            for m_name in model_names:
                for c in multi_codec:
                    out_path = os.path.join(self.conf["OUT_DB"], m_name + str(c), ffolder)
                    if not os.path.exists(os.path.join(out_path, fname)):
                        if c not in outputs_todo[m_name]:
                            outputs_todo[m_name].append(c)
        
        # Check if there's anything to do
        need_processing = False
        for m_name in outputs_todo:
            if outputs_todo[m_name]:
                need_processing = True
                
        if not need_processing:
            print("All outputs already exist. Nothing to do.")
            return
            
        # Process files in batches to better utilize GPU
        batch_size = self.conf["BATCH_SIZE"]
        file_batches = [paths_to_process[i:i+batch_size] for i in range(0, len(paths_to_process), batch_size)]
        
        # Before processing, move models to the correct device
        for model in models:
            if hasattr(model, 'to') and callable(getattr(model, 'to')):
                model.to(self.conf["DEVICE"])
                
        for i, batch in enumerate(tqdm(file_batches, desc="Processing batches")):
            self.process_batch(batch, models, model_names, multi_codec, outputs_todo)
            
            if i % 10 == 0:  # Periodically clear CUDA cache
                torch.cuda.empty_cache()
