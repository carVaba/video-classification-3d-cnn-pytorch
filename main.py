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

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    #Usage -> https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console?page=1&tab=votes#tab-top
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

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
    l , i = len(input_files) , 0
    printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=100)
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            subprocess.call(f"mkdir {process_video_folder}/{input_file}", shell=True)
            subprocess.call(f"ffmpeg -i {video_path} -loglevel quiet {process_video_folder}/{input_file}/image_%05d.jpg",
                            shell=True)
            try:
                input_file = f"{process_video_folder}/{input_file}"
                result = classify_video(input_file, input_file, class_names, model, opt)
                outputs.append(result)
            except Exception as ex:
                print(ex)
                #print(f"Error for the file {input_file}")
            #subprocess.call('rm -rf tmp', shell=True)
        else:
            print(f"{input_file} does not exist")

        printProgressBar(i + 1, l, prefix='Progress:',
                             suffix='Complete', length=100)
        i += 1

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)
