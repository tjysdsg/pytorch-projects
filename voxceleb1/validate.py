import os
import argparse
import kaldiio
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
import numpy as np
from config import PROJECT_ROOT_DIR


def compute_eer(y_pred, y):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    thre = threshold[idx]
    return eer, thre


def cosine_scoring(embd1s, embd2s):
    scores = []
    for embd1, embd2 in zip(embd1s, embd2s):
        # ! No errors here. Google scipy.spatial.distance.cosine for details.
        score = 1 - cosine(embd1, embd2) / 2
        scores.append(score)
    return scores


def main(args):
    trials = [x.split() for x in open(os.path.join(args.data, 'trials'))]
    utt1s = [x[0] for x in trials]
    utt2s = [x[1] for x in trials]
    if len(trials[0]) == 3:
        tar2int = {'nontarget': 0, 'target': 1}
        target = [tar2int[x[2]] for x in trials]
    else:
        target = None

    embd_scp = os.path.join(args.data, 'embedding.scp')
    with kaldiio.ReadHelper(f'scp:{embd_scp}') as reader:
        utt2embd = {utt: embd for utt, embd in reader}

    embd1s = [utt2embd[utt] for utt in utt1s]
    embd2s = [utt2embd[utt] for utt in utt2s]

    scores = cosine_scoring(embd1s, embd2s)
    score_path = os.path.join(args.data, 'scores.txt')
    np.savetxt(score_path, scores, fmt='%.4f')

    if target is not None:
        eer, threshold = compute_eer(scores, target)
        print("EER: {:.2f}%".format(eer * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speaker Verification Model Validation')
    parser.add_argument('--data', type=str, default=os.path.join(PROJECT_ROOT_DIR, 'voxceleb1', 'data', 'vox1_test'),
                        help='data directory of inputs and outputs.')
    args = parser.parse_args()

    for f in ['embedding.scp', 'trials']:
        fpath = os.path.join(args.data, f)
        assert os.path.isfile(fpath), "No such file: %s" % f

    main(args)
