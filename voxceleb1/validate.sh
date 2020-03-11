#!/bin/bash
set -e 

data=data/vox1_test
saved_dir=exp/resnet34c16_200to400_vox12_b512/chkpt
config=$saved_dir/config.json
resume=$saved_dir/chkpt_050.pth

python infer.py -c $config -r $resume $data
python validate.py $data
