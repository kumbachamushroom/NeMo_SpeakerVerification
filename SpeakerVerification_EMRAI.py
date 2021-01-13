import os
import glob
import json
import warnings
import torch
import pandas as pd
import numpy as np
from math import floor

import pytorch_lightning as pl
from omegaconf.listconfig import ListConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import hydra
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel

import nemo
import nemo.collections.asr as nemo_asr
def label_frames(label_path, window_size, step_size):
    '''
    :param label_path: path to .lab file
    :param window_size: frame_windows size (.lab is only 10ms)
    :param step_size: step_size for sliding window
    :return: frame label dataframes of shape speakers x frames
    '''
    def get_common_label(List):
        return max(set(List), key=List.count)
    labels = [int(line) for line in open(label_path)]
    duration = len(labels)*0.01
    n_increments = floor((duration - window_size)/ step_size)
    frame_list = []
    for i in range(n_increments + 2):
        start_time = i * step_size
        stop_time = start_time + window_size
        frame_list.append((start_time, stop_time))
    speaker_list = ['Agent', 'Caller']
    for frame in frame_list:
        start_time, stop_time = int(frame[0]/0.01), int(frame[1]/0.01)
        try:
            frame_label = get_common_label(labels[start_time:stop_time])
        except:
            None
        if frame_label == 0:
            frame_list.remove(frame)
    speaker_df = pd.DataFrame(columns=speaker_list, data=np.zeros(shape=(len(frame_list), len(speaker_list)), dtype=int))
    for i, frame in enumerate(frame_list):
        start_time, stop_time = int(frame[0]/0.01), int(frame[1]/0.01)
        try:
            frame_label = get_common_label(labels[start_time:stop_time])
        except:
            None
        if frame_label != 0:
            speaker_df.iloc[i, frame_label-1] = 1
    return frame_list, speaker_df

def compute_steps(frame_length, overlap, audio_path):
    '''
    Computes start/end times for each frame in audio file
    :param step_length: length of each frame (seconds)
    :param overlap: overlap between frames (seconds)
    :param audio_directory: path to audio file
    :return: list of tuples in form (start, duration)
    '''


def write_target_manifest(audio_path, length, manifest_file, agent):
    if os.path.exists(os.path.join(os.getcwd(), 'manifest_files', manifest_file)):
        os.remove(os.path.join(os.getcwd(), 'manifest_files', manifest_file))
    with open(os.path.join(os.getcwd(), 'manifest_files', manifest_file), 'a') as outfile:
        meta = {"audio_filepath":audio_path, "duration":10, "label":'agent'}
        json_str = json.dumps(meta)
        outfile.write(json_str)
        outfile.write("\n")
    print("Created target-speaker manifest file...")


def write_track_manifest(audio_path, frame_list, manifest_file):
    if os.path.exists(os.path.join(os.getcwd(), 'manifest_files', manifest_file)):
        os.remove(os.path.join(os.getcwd(), 'manifest_files', manifest_file))
    with open(os.path.join(os.getcwd(), 'manifest_files', manifest_file), 'a') as outfile:
        for i in range(len(frame_list)):
            start, stop = round(frame_list[i][0],1), round(frame_list[i][1],1)
            meta = {"audio_filepath":audio_path, "offset":start, "duration":stop, "label":'agent'}
            json_str = json.dumps(meta)
            outfile.write(json_str)
            outfile.write('\n')



##def get_embedding_from_pickle(emb_pckl, target):
#    '''
#    This function unpickles the pickle object returned by the SpeakerNet Embedding and returns the desired embedding
#    :param emb_pckl: serialized key-pair dict object
#    :param target: name of target_embedding -> path@to@directory
#    :return: 512-dimensional embedding -> numpy array
#    '''
    


seed_everything(42)

@hydra.main(config_path='SpeakerVerification_EMRAI.yaml')
def main(cfg: DictConfig) -> None:
    os.chdir('/home/lucas/PycharmProjects/NeMo_SpeakerVerification')
    cuda = 1 if torch.cuda.is_available() else 0
    model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name='SpeakerNet_verification')
    #SpeakerNet_verification = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="SpeakerNet_verification")
    audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)[:cfg.audio.num_target_tracks]
    for track in audio_tracks:
        agent=track[track.find('-')+1:track.find('.')]
        agent_samples = glob.glob(cfg.audio.verification_path+agent+'.wav', recursive=True)
        if len(agent_samples) > 0:
            write_target_manifest(audio_path=agent_samples[0], length=cfg.audio.verification_length, manifest_file='target_test.json',agent=agent)
            #SpeakerNet_verification.setup_test_data(write_target_manifest(audio_path=agent_samples[0], length=cfg.audio.verification_length, manifest_file='target.json',agent=agent))
            #trainer = pl.Trainer(gpus=cuda, accelerator=None)
            #trainer.test(SpeakerNet_verification)
        else:
            warnings.warn('Verification audio for {} not found '.format(agent))
    test_config = OmegaConf.create(dict(
        manifest_filepath = os.path.join(os.getcwd(), 'manifest_files', 'target_test.json'),
        sample_rate = 16000,
        labels = None,
        batch_size = 1,
        shuffle=False,
        time_length = 3,
        embedding_dir=os.path.join(os.getcwd(),'embeddings')
    ))
    model.setup_test_data(test_config)
    trainer = pl.Trainer(gpus=cuda)
    trainer.test(model)
    for window_length in cfg.audio.window_length:
        for step_length in cfg.audio.step_length:
            for track in audio_tracks:
                label_path = track[track.rfind('/')+1:track.find('.wav')]+'.labs'
                frame_list, speaker_df = label_frames(label_path=os.path.join(cfg.audio.label_path, label_path),
                                                      window_size=window_length,
                                                      step_size=float(window_length*step_length))
                write_track_manifest(audio_path=track, frame_list=frame_list, manifest_file='track_manifest.json')
                test_config = OmegaConf.create(dict(
                    manifest_filepath = os.path.join(os.getcwd(), 'manifest_files', 'track_manifest.json'),
                    sample_rate = 16000,
                    labels = None,
                    batch_size = 1,
                    shuffle = False,
                    embedding_dir = os.path.join(os.getcwd(),'embeddings')
                ))
                model.setup_test_data(test_config)
                trainer = pl.Trainer(gpus=cuda)
                trainer.test(model)



if __name__ == '__main__':
    main()