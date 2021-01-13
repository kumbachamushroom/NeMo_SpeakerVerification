import os
import glob
import json

import pytorch_lightning as pl
from omegaconf.listconfig import ListConfig
from pytorch_lightning import seed_everything

import nemo
import nemo.collections.asr as nemo_asr

SpeakerNet_verification = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="SpeakerNet_verification")

def compute_steps(frame_length, overlap, audio_path):
    '''
    Computes start/end times for each frame in audio file
    :param step_length: length of each frame (seconds)
    :param overlap: overlap between frames (seconds)
    :param audio_directory: path to audio file
    :return: list of tuples in form (start, duration)
    '''


def write_manifest(step_length, overlap, audio_path, manifest_file):
