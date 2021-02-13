import os
import glob
import subprocess

load_dir = '/home/lucas/PycharmProjects/Data/EMRAI/dev_other/wavs/*'
save_dir = '/home/lucas/PycharmProjects/Data/EMRAI/dev_other/wav49s/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

track_list = glob.glob(load_dir, recursive=True)

for track in track_list:
    name = track[track.rfind('/')+1:]
    command = 'sox {} -r 8000 -c1 {}'.format(track, save_dir+name)
    subprocess.call(command, shell=True)
    #sox 102884.wav -r 8000 -c1 102884_enc.wav
