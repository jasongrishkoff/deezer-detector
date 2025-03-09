"""
Modified Pipeline looper to work with a sample of files
"""

import os
from time import time

import numpy as np
import soxr
import torch
import torchaudio
from tqdm import tqdm

default_conf = {
    "DEVICE": "cpu",
    "DB_PATH": "/",
    "OUT_DB": "/",
    "BR_PATH": "/",
    "DATA_SR": 44100,
    "SR": 44100, # target sr
    "TO_MONO": False,
    "AUGMENT": False,
    "MIN_DURATION": 3, # seconds
    "MAX_DURATION": 40,
    "VERBOSE": False,
    "BITRATE": 320,  # Default bitrate when no bitrate data is available
}


class SamplePipeline:
    def __init__(self, paths, conf = {}):
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
        self.bug_mode = False


    def clock(self, ops_name):
        if self.conf["VERBOSE"]:
            current_time = time()
            print("{}: {:.2f}s".format(ops_name, current_time - self.global_t ))
            self.global_t = current_time


    def run_loop(self, models, model_names, multi_codec = [], has_cpu_preprocess=False, max_files=None):
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

        for fpath in tqdm(paths_to_process):
            fname = os.path.basename(fpath)
            ffolder = os.path.basename(os.path.dirname(fpath))

            # check skip already done
            multi_codec_todo = []
            for m_name in model_names:
                for c in multi_codec:
                    out_path = os.path.join(self.conf["OUT_DB"], m_name + str(c), ffolder)
                    if not os.path.exists(os.path.join(out_path, fname)):
                        os.makedirs(out_path, exist_ok = True)
                        multi_codec_todo.append(c)

            if len(multi_codec_todo) == 0:
                continue  # Skip if all outputs already exist

            # open audio
            try:
                audio_raw, sr = torchaudio.load(fpath)
                if sr != self.conf["SR"]:
                    print("Resampling {}: {} -> {}".format(fpath, sr, self.conf["SR"]))
                    audio_raw_rs = soxr.resample(audio_raw.T, sr, self.conf["SR"]).T
                    audio_raw = torch.Tensor(audio_raw_rs)

                audio_raw = torch.squeeze(audio_raw)
                if audio_raw.shape[-1] < self.conf["SR"] * self.conf["MIN_DURATION"]:
                    print("Track {} < min duration, skipping".format(fpath))
                    continue
                if len(audio_raw.shape) == 1:  # mono
                    audio_raw = torch.stack([audio_raw, audio_raw], 0)
                if audio_raw.shape[-1] > self.conf["SR"] * self.conf["MAX_DURATION"]:
                    audio_raw = audio_raw[:,:self.conf["SR"] * self.conf["MAX_DURATION"]]
            except Exception as err:
                print("Track {} failed: [{}] {}".format(fpath, type(err), err))
                continue

            if not has_cpu_preprocess:
                audio_raw = audio_raw.to(self.conf["DEVICE"])
            
            # Get bitrate from dictionary or use default
            audio_br = self.bitrates.get(fname, self.conf["BITRATE"])
            if isinstance(audio_br, (int, float)):
                audio_br = min(int(audio_br), 320)
            else:
                audio_br = self.conf["BITRATE"]
                
            self.clock("opening")

            # run autoencoding and save result
            for m, m_name in zip(models, model_names):
                if len(multi_codec_todo) == 0:
                    continue

                with torch.no_grad():
                    if not using_multi:
                        audio_rebuilt = m.autoencode(audio_raw)
                        audios_rebuilt = [audio_rebuilt.to("cpu")]
                    else:
                        audios_rebuilt = m.autoencode_multi(audio_raw, multi_codec_todo)
                        audios_rebuilt = [audio.to("cpu") for audio in audios_rebuilt]

                self.clock("autoencode")

                for audio_rebuilt, c in zip(audios_rebuilt, multi_codec_todo):
                    out_path = os.path.join(self.conf["OUT_DB"], m_name + str(c), ffolder)
                    os.makedirs(out_path, exist_ok=True)
                    out_file = os.path.join(out_path, fname)
                    
                    torchaudio.save(out_file,
                            audio_rebuilt,
                            sample_rate = self.conf["SR"],
                            channels_first = True,
                            format="mp3", 
                            )

                    self.clock("saved")
