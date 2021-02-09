import os
import glob
import json

import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import hydra
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel

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

import pickle

def get_all_speaker(audio_dir):
    tracks = glob.glob('/home/lucas/PycharmProjects/MetricEmbeddingNet/temp_files/with_vad/*', recursive=True)
    return tracks

def write_manifest(speaker_list):
    with open(os.path.join(os.getcwd(),'manifest_files','libri_test.json')) as outfile:
        for track in speaker_list:
            for i in range(3):
                meta = {"audio_filepath":track, "offset":i*8,"duration":8}

@hydra.main(config_path='SpeakerVerification_LibriSpeech.yaml')
def main(cfg: DictConfig) -> None:
    os.chdir('/home/lucas/PycharmProjects/NeMo_SpeakerVerification')
    speaker_list = get_all_speaker(audio_dir=cfg.audio.path)
    print(speaker_list)

if __name__ == "__main__":
    main()


