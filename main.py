import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.n_classes = 400
    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    outputs = []
    process_video_folder = "videoPRO"
    if os.path.exists(process_video_folder):
        subprocess.call(f"rm -rf {process_video_folder}" , shell=True)
    os.mkdir(process_video_folder)
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            print(video_path)
            subprocess.call(f"mkdir {process_video_folder}/{input_file}", shell=True)
            subprocess.call(f"ffmpeg -i {video_path} -loglevel quiet {process_video_folder}/{input_file}/image_%05d.jpg",
                            shell=True)
            try:
                result = classify_video(input_file, input_file, class_names, model, opt)
                outputs.append(result)
            except:
                print(f"Error for the file {input_file}")
            #subprocess.call('rm -rf tmp', shell=True)
        else:
            print(f"{input_file} does not exist")

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)
