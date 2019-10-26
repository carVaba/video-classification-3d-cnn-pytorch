import subprocess

subprocess.call("python main.py --input ./input \
--video_root ./videos \
--output ./outputSampl11Over10.json \
--model ./resnext-101-kinetics.pth \
--resnet_shortcut B \
--model_name resnext  \
--model_depth 101 \
--mode feature \
--overlapping 1 \
--sample_duration 11", shell = True)
