import subprocess

subprocess.call("python main.py --input ./input --video_root ./videos --output ./outputSampl11Over10.json --model ./resnet-34-kinetics.pth --mode feature --overlapping 10 --sample_duration 11", shell = True)
