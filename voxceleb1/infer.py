import os
import json
import GPUtil
import argparse
import torch
from tqdm import tqdm
import kaldiio
import sys
from config import PROJECT_ROOT_DIR

sys.path.insert(0, PROJECT_ROOT_DIR)

import voxceleb1.dataset as dataset
from torch.utils.data import DataLoader
import voxceleb1.module.model as module_model


def get_instance(module, cfs, *args):
    return getattr(module, cfs['type'])(*args, **cfs['args'])


def set_device(n_gpu):
    if n_gpu > 0:
        device = 'cuda'
        deviceIDs = GPUtil.getAvailable(limit=n_gpu, maxMemory=0.9, maxLoad=0.9)
        assert deviceIDs != [], "n_gpu > 0, but no GPUs available!"
        print("Use GPU:", deviceIDs)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, deviceIDs))
    else:
        print("Use CPU")
        device = 'cpu'
    return device


def main(config, args):
    device = set_device(1)

    # g-vector extractor
    model = get_instance(module_model, config['model'])
    chkpt = torch.load(args.resume)
    try:
        model.load_state_dict(chkpt['model'])
    except:
        model.load_state_dict(chkpt)
    model = model.to(device)

    config['dataset']['args']['wav_scp'] = os.path.join(args.data, 'wav.scp')
    config['dataset']['args']['utt2spk'] = None
    testset = get_instance(dataset, config['dataset'])
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )

    model.eval()
    utt2embd = {}
    for i, (utt, data) in enumerate(tqdm(testloader, ncols=80)):
        utt = utt[0]
        data = data.float().to(device)
        with torch.no_grad():
            embd = model.extractor(data)
        embd = embd.squeeze(0).cpu().numpy()
        utt2embd[utt] = embd

    embd_wfile = 'ark,scp:{0}/embedding.ark,{0}/embedding.scp'.format(args.data)
    with kaldiio.WriteHelper(embd_wfile) as writer:
        for utt, embd in utt2embd.items():
            writer(utt, embd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speaker Verification Inference')
    parser.add_argument('-c', '--config', type=str, required=True, help='config file path')
    parser.add_argument('-r', '--resume', type=str, required=True, help='path to latest checkpoint')
    parser.add_argument('--data', type=str, default=os.path.join(PROJECT_ROOT_DIR, 'voxceleb1', 'data', 'vox1_test'),
                        help='data directory of inputs and outputs.')
    args = parser.parse_args()
    # Read config of the whole system.
    with open(args.config) as rfile:
        config = json.load(rfile)

    main(config, args)
