import os
import glob
import json
import warnings
import torch

import pytorch_lightning as pl
from omegaconf.listconfig import ListConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import hydra

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


def write_target_manifest(audio_path, length, manifest_file):
    if os.path.exists(os.path.join(os.getcwd(), 'manifest_files', manifest_file)):
        os.remove(os.path.join(os.getcwd(), 'manifest_files', manifest_file))
    with open(manifest_file, 'w') as outfile:
        meta = {"audio_filepath":audio_path, "duration":length}
        json.dump(meta, outfile)
        outfile.write("\n")
    return OmegaConf.create(dict(
        manifest_filepath = os.path.join(os.getcwd(),'manifest_files',manifest_file),
        sample_rate = 16000,
        labels = None,
        batch_size = 1,
        shuffle = False,
        time_length = length,
        embedding_dir = os.path.join(os.getcwd(),'embeddings')
    ))



@hydra.main(config_path='SpeakerVerification_EMRAI.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    cuda = 1 if torch.cuda.is_available() else 0
    SpeakerNet_verification = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="SpeakerNet_verification")
    audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)[:cfg.audio.num_target_tracks]
    for track in audio_tracks:
        agent=track[track.find('-')+1:track.find('.')]
        agent_samples = glob.glob(cfg.audio.verification_path+agent+'.wav', recursive=True)
        if len(agent_samples) > 0:
            SpeakerNet_verification.setup_test_data(write_target_manifest(audio_path=agent_samples[0], length=cfg.audio.verification_length, manifest_file='target.json'))
            trainer = pl.Trainer(gpus=cuda, accelerator=None)
            trainer.test(SpeakerNet_verification)
        else:
            warnings.warn('Verification audio for {} not found '.format(agent))


if __name__ == '__main__':
    main()