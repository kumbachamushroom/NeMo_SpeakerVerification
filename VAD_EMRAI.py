import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from math import floor
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import copy
import pickle

import pytorch_lightning as pl
from omegaconf.listconfig import ListConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import hydra
import nemo
import nemo.collections.asr as nemo_asr
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from torch.utils.data import DataLoader


def create_track_manifest(cfg):
    tracks = glob.glob(cfg.audio.input_dir, recursive=True)
    if cfg.audio.num_tracks > -1:
        tracks = tracks[:cfg.audio.num_tracks]

    def get_length(track):
        name = track[track.rfind('/')+1:track.rfind('.')]+'.labs'
        duration = round(len([line for line in open(cfg.audio.label_dir+name)])*0.01,2)
        return duration

    if os.path.exists(cfg.audio.manifest_path):
        os.remove(cfg.audio.manifest_path)
    with open(cfg.audio.manifest_path,'a')as outfile:
        for track in tracks:
            duration = get_length(track)
            json_str = json.dumps({"audio_filepath":track,
                                   "offset":0.0,
                                   "duration":duration,
                                   "label":"EMRAI_dev_other",
                                   })
            outfile.write(json_str)
            outfile.write('\n')


#def create_file_manifest()

class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = signal.astype(np.float32) / 32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1

@hydra.main(config_path='MatchBoxNet_VAD.yaml')
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda:0')
    model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name='MatchboxNet-VAD-3x2')
    cfg_vad = copy.deepcopy(model._cfg)
    print(OmegaConf.to_yaml(cfg_vad))
    model.preprocessor = model.from_config_dict(cfg_vad.preprocessor)
    model.eval()
    model.to(device)

    # simple data layer to pass audio signal


    data_layer = AudioDataLayer(sample_rate=cfg.train_ds.sample_rate)
    data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

    def infer_signal(model, signal):
        data_layer.set_signal(signal)
        batch = next(iter(data_loader))
        audio_signal, audio_signal_len = batch
        audio_signal, audio_signal_len = audio_signal.to(model.device), audio_signal_len.to(model.device)
        logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        return logits

    class FrameVAD:

        def __init__(self, model_definition,
                     frame_len=2, frame_overlap=2.5,
                     offset=10):
            '''
            Args:
              frame_len: frame's duration, seconds
              frame_overlap: duration of overlaps before and after current frame, seconds
              offset: number of symbols to drop for smooth streaming
            '''
            self.vocab = list(model_definition['labels'])
            self.vocab.append('_')

            self.sr = model_definition['sample_rate']
            self.frame_len = frame_len
            self.n_frame_len = int(frame_len * self.sr)
            self.frame_overlap = frame_overlap
            self.n_frame_overlap = int(frame_overlap * self.sr)
            timestep_duration = model_definition['AudioToMFCCPreprocessor']['params']['window_stride']
            for block in model_definition['JasperEncoder']['params']['jasper']:
                timestep_duration *= block['stride'][0] ** block['repeat']
            self.buffer = np.zeros(shape=2 * self.n_frame_overlap + self.n_frame_len,
                                   dtype=np.float32)
            self.offset = offset
            self.reset()

        def _decode(self, frame, offset=0):
            assert len(frame) == self.n_frame_len
            self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
            self.buffer[-self.n_frame_len:] = frame
            logits = infer_signal(vad_model, self.buffer).cpu().numpy()[0]
            decoded = self._greedy_decoder(
                logits,
                self.vocab
            )
            return decoded

        @torch.no_grad()
        def transcribe(self, frame=None):
            if frame is None:
                frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
            if len(frame) < self.n_frame_len:
                frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
            unmerged = self._decode(frame, self.offset)
            return unmerged

        def reset(self):
            '''
            Reset frame_history and decoder's state
            '''
            self.buffer = np.zeros(shape=self.buffer.shape, dtype=np.float32)
            self.prev_char = ''

        @staticmethod
        def _greedy_decoder(logits, vocab):
            s = []
            if logits.shape[0]:
                probs = torch.softmax(torch.as_tensor(logits), dim=-1)
                probas, preds = torch.max(probs, dim=-1)
                s = [preds.item(), str(vocab[preds]), probs[0].item(), probs[1].item(), str(logits)]
            return s

    STEP_LIST = [0.01, 0.01, ]
    WINDOW_SIZE_LIST = [0.31, 0.15, ]

if __name__ == '__main__':
    main()
