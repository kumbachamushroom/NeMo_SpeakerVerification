import os
import glob
import json
import warnings
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

import pytorch_lightning as pl
from omegaconf.listconfig import ListConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import hydra
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel

#Pyannote tools used for DER, merging frames, reading rttms etc..
from pyannote.database.util import load_rttm
from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment


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




def write_target_manifest(audio_path, length, manifest_file, agent):
    with open(os.path.join(os.getcwd(), 'manifest_files', manifest_file), 'a') as outfile:
        meta = {"audio_filepath":audio_path, "duration":length, "label":'agent'}
        json_str = json.dumps(meta)
        outfile.write(json_str)
        outfile.write("\n")
    print("Created target-speaker manifest file...")


def write_track_manifest(audio_path, frame_list, manifest_file, window_length, step_length):
    #if os.path.exists(os.path.join(os.getcwd(), 'manifest_files', manifest_file)):
    #    os.remove(os.path.join(os.getcwd(), 'manifest_files', manifest_file))
    with open(os.path.join(os.getcwd(), 'manifest_files', manifest_file), 'a') as outfile:
        for i in range(len(frame_list)):
            start, stop = round(frame_list[i][0],1), round(frame_list[i][1],1)
            meta = {"audio_filepath":audio_path, "offset":start, "duration":window_length, "label":'agent',
                    'window_length':window_length, 'step_length':step_length}
            json_str = json.dumps(meta)
            outfile.write(json_str)
            outfile.write('\n')




def cluster_embeddings(agent, track, window_length, step_length, track_embedding):
    #track_embedding = []
    #track = track.split('/')[-1]
    #indices = [track_manifest.index(item) for item in track_manifest if item['audio_filepath'] == track and item["duration"] == window_length and item["step_length"] == step_length]
    #with open(os.path.join(os.getcwd(),'embeddings','track_manifest_embeddings.pkl'), 'rb') as f:
    #    data = pickle.load(f).items()
    #    track_embedding = [emb for _, emb in data][min(indices):max(indices)+1]
        #for name, emb in data:
        #    if track in name:
        #        track_embedding.append(emb)

    with open(os.path.join(os.getcwd(),'embeddings','target_embeddings.pkl'), 'rb') as f:
        data = pickle.load(f).items()
        for name, emb in data:
            #print(agent, name)
            if agent in name:
                target_embedding = emb

    # Initialise cluster and fit
    kmeans_cluster = KMeans(n_clusters=2, random_state=5)
    #kmeans_cluster = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average')
    kmeans_cluster.fit_predict(X=track_embedding)

    #PCA cluster plot
    cluster_PCA(track_embedding, kmeans_cluster.labels_)

    # Get average embeddings of each cluster
    cluster1 = [track_embedding[i] for i,j in enumerate(kmeans_cluster.labels_) if (j == 0)]
    cluster2 = [track_embedding[i] for i,j in enumerate(kmeans_cluster.labels_) if (j == 1)]
    #print('Lengths of cluster 1 {} and 2 {}'.format(len(cluster1), len(cluster2)))
    cluster1_avg = np.mean(cluster1, axis=0)
    cluster2_avg = np.mean(cluster2, axis=0)

    # Compute cluster distances from target embedding
    cluster1_dist = cdist(cluster1_avg.reshape(1,-1), target_embedding.reshape(1,-1), metric='cosine')[0][0]
    cluster2_dist = cdist(cluster2_avg.reshape(1,-1), target_embedding.reshape(1,-1), metric='cosine')[0][0]
    target_cluster = 0 if cluster1_dist < cluster2_dist else 1
    #print('CLUSTER 1 DIST IS {} AND CLUSTER 2 DIST IS {}'.format(cluster1_dist, cluster2_dist))

    outputs = []
    for label in kmeans_cluster.labels_:
        outputs.append(1) if label == target_cluster else outputs.append(2)
    return outputs

def cluster_PCA(cluster_outputs, labels):
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(cluster_outputs)
    pca_one = pca_results[:,0]
    pca_two = pca_results[:,1]
    pca_three = pca_results[:,2]
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(xs = pca_one,ys=pca_two, zs=pca_three,c=labels, cmap='tab10')
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    ax.set_title('PCA-3 decomposition for 512-dimensional embedding')
    plt.show()

    df = pd.DataFrame()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(cluster_outputs)
    df['tsne-one'] = tsne_results[:,0]
    df['tsne-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='tsne-one',
        y='tsne-two',
        palette=sns.color_palette("hls",10),
        data=df,
        legend="full",
        alpha=0.3
    )


def get_performance_metrics(speaker_df, outputs):
    agent_labels = np.array(speaker_df.Agent.tolist())
    caller_labels = np.array(speaker_df.Caller.tolist())



    agent_output_mask = agent_labels*outputs
    caller_output_mask = caller_labels*outputs
    #print(agent_output_mask)
    #print(caller_output_mask)

    agent_coverage = np.count_nonzero(agent_output_mask == 1)/np.count_nonzero(agent_output_mask != 0)
    caller_coverage = np.count_nonzero(caller_output_mask == 2)/np.count_nonzero(caller_output_mask != 0)

    agent_purity = np.count_nonzero(agent_output_mask == 1)/np.count_nonzero(outputs == 1)
    caller_purity = np.count_nonzero(caller_output_mask == 2)/np.count_nonzero(outputs == 2)

    return (agent_coverage+caller_coverage)/2, (agent_purity+caller_purity)/2

def merge_frames(outputs, frame_list):
    annotation = Annotation()
    for speaker_num in range(1,3):
        seg_start = 0
        seg_end = 0
        index = 0
        smooth_segment = Segment(start=0, end=0)
        skip = 0
        for i, label in enumerate(outputs):
            if (label == speaker_num) and (seg_start == 0) and (seg_end == 0):
                if (skip != 0) and (skip < 1):
                    try:
                        del annotation[smooth_segment]
                        seg_start = float(smooth_segment.start)
                        seg_end = float(smooth_segment.end)
                        skip = 0
                    except:
                        None
                else:
                    seg_start = float(frame_list[i][0])
                    seg_end = float(frame_list[i][1])
                    index = i
            elif (label != speaker_num) and (seg_end > 0):
                annotation[Segment(start=seg_start, end = seg_end)] = speaker_num
                smooth_segment = Segment(start=seg_start, end=seg_end)
                skip = 1
                seg_start = 0
                seg_end = 0
            elif (seg_end > 0) and ((i - index) == 1):
                index = i
                seg_end = frame_list[i][1]
                #print('step length away')
            elif (seg_end > 0) and ((i - index) > 1):
                annotation[Segment(start=seg_start, end=seg_end)] = speaker_num
                seg_start = float(frame_list[i][0])
                seg_end = float(frame_list[i][1])
                index = i
            elif (label != speaker_num) and (seg_end == 0):
                skip = skip + 1
    return annotation

def get_der(cfg, rttm, output_annotations):
    metric = DiarizationErrorRate(skip_overlap=True, collar=cfg.audio.collar)
    groundtruth = load_rttm(rttm)[rttm[rttm.rfind('/')+1:rttm.find('.')]]
    der = metric(groundtruth, output_annotations, detailed=False)
    return der

seed_everything(42)

@hydra.main(config_path='SpeakerVerification_EMRAI.yaml')
def main(cfg: DictConfig) -> None:
    os.chdir('/home/lucas/PycharmProjects/NeMo_SpeakerVerification')
    cuda = 1 if torch.cuda.is_available() else 0
    model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name='SpeakerNet_verification')
    #SpeakerNet_verification = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="SpeakerNet_verification")
    if cfg.audio.num_target_tracks > -1:
        audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)[:cfg.audio.num_target_tracks]
    else:
        audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)

    #Text files for logging
    der_log = open('/home/lucas/PycharmProjects/NeMo_SpeakerVerification/Txt_outs/der_cluster_noiseless.txt', 'w')

    if os.path.exists(os.path.join(os.getcwd(), 'manifest_files', 'target.json')):
        os.remove(os.path.join(os.getcwd(), 'manifest_files', 'target.json'))


    #Write target-speaker manifest files and check to see that all audio files have matching target files
    for track in audio_tracks:
        agent=track[track.find('-')+1:track.find('.')]
        agent_samples = glob.glob(cfg.audio.verification_path+agent+'.wav', recursive=True)
        if len(agent_samples) > 0:
            write_target_manifest(audio_path=agent_samples[0], length=cfg.audio.verification_length, manifest_file='target.json',agent=agent)
            # write_track_manifest(audio_path=track, frame_list=frame_list, manifest_file='track_manifest.json')
            #model.setup_test_data(write_target_manifest(audio_path=agent_samples[0], length=cfg.audio.verification_length, manifest_file='target.json',agent=agent))
            #trainer = pl.Trainer(gpus=cuda, accelerator=None)
            #trainer.test(model)
        else:
            warnings.warn('Verification audio for {} not found '.format(agent))
    test_config = OmegaConf.create(dict(
        manifest_filepath = os.path.join(os.getcwd(), 'manifest_files', 'target.json'),
        sample_rate = 16000,
        labels = None,
        batch_size = 1,
        shuffle=False,
        embedding_dir='./'#os.path.join(os.getcwd(),'embeddings')
    ))
    model.setup_test_data(test_config)
    trainer = pl.Trainer(gpus=cuda)
    trainer.test(model)

    test_config = OmegaConf.create(dict(
        manifest_filepath=os.path.join(os.getcwd(), 'manifest_files', 'track_manifest.json'),
        sample_rate=16000,
        labels=None,
        batch_size=16,
        shuffle=False,
        embedding_dir='./' ,
        num_workers = 4# os.path.join(os.getcwd(),'embeddings')
    ))

    if os.path.exists(os.path.join(os.getcwd(), 'manifest_files', 'track_manifest.json')):
        os.remove(os.path.join(os.getcwd(), 'manifest_files', 'track_manifest.json'))

    for window_length in cfg.audio.window_length:
        for step_length in cfg.audio.step_length:
            for track in audio_tracks:
                agent = track[track.find('-') + 1:track.find('.')]
                agent_samples = glob.glob(cfg.audio.verification_path + agent + '.wav', recursive=True)
                rttm = glob.glob(cfg.audio.rttm_path + track[track.rfind('/') + 1:track.rfind('.')] + '.rttm',
                                 recursive=False)[0]
                #print(agent_samples)
                if len(agent_samples) > 0:
                    label_path = track[track.rfind('/')+1:track.find('.wav')]+'.labs'
                    frame_list, speaker_df = label_frames(label_path=os.path.join(cfg.audio.label_path, label_path),
                                                      window_size=window_length,
                                                      step_size=float(window_length*step_length))
                    write_track_manifest(audio_path=track, frame_list=frame_list, manifest_file='track_manifest.json', window_length=window_length, step_length=step_length)
    model.setup_test_data(test_config)
    trainer = pl.Trainer(gpus=cuda)
    trainer.test(model)
    track_manifest = [json.loads(line.replace('\n', '')) for line in
                      open(os.path.join(os.getcwd(), 'manifest_files', 'track_manifest.json'))]
    with open(os.path.join(os.getcwd(),'embeddings','track_manifest_embeddings.pkl'), 'rb') as f:
        data = pickle.load(f).items()
        all_track_embeddings = [emb for _, emb in data]
    for window_length in cfg.audio.window_length:
        for step_length in cfg.audio.step_length:
            for track in audio_tracks:
                agent = track[track.find('-') + 1:track.find('.')]
                agent_samples = glob.glob(cfg.audio.verification_path + agent + '.wav', recursive=True)
                rttm = glob.glob(cfg.audio.rttm_path + track[track.rfind('/') + 1:track.rfind('.')] + '.rttm',
                                 recursive=False)[0]
                # print(agent_samples)
                if len(agent_samples) > 0:
                    label_path = track[track.rfind('/') + 1:track.find('.wav')] + '.labs'
                    frame_list, speaker_df = label_frames(label_path=os.path.join(cfg.audio.label_path, label_path),
                                                          window_size=window_length,
                                                          step_size=float(window_length * step_length))
                    indices = [track_manifest.index(item) for item in track_manifest if
                               item['audio_filepath'] == track and item["duration"] == window_length and item[
                                   "step_length"] == step_length]
                    print(indices)
                    embedddings = all_track_embeddings[min(indices):max(indices)+1]
                    cluster_outputs = cluster_embeddings(agent=agent, track=track, window_length=window_length, step_length=step_length, track_embedding=embedddings)
                    #print(len(cluster_outputs))
                    #print(speaker_df.describe())
                    coverage, purity = get_performance_metrics(speaker_df, np.array(cluster_outputs))
                    print("The results for {} -> Coverage {} / Purity {}".format(track, coverage, purity))
                    annotation = merge_frames(outputs=cluster_outputs, frame_list=frame_list)
                    der = get_der(cfg=cfg, rttm=rttm, output_annotations=annotation)
                    print('THE DER IS {}'.format(der))
                    der_log.write('{} \t {} \t {} \t {} \t {} \t {} \n'.format(track, window_length, step_length, coverage,
                                                                     purity, der))

    der_log.close()

if __name__ == '__main__':
    main()