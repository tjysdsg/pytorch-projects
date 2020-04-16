#!/usr/bin/env bash
python voxceleb1/infer.py -c voxceleb1/config.json -r models/voxceleb1/chkpt/chkpt_100.pth
python voxceleb1/validate.py
