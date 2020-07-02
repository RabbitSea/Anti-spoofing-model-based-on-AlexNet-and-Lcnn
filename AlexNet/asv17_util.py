import h5py
import numpy as np

from AlexNet import eval_metrics as em

label_encoder = {'genuine': 0, 'spoof': 1}

'''计算eer'''


def compute_eer17_from_file(score_file):
    score_data = np.genfromtxt(score_file, dtype=str)
    # utterances = score_data[:, 0]
    # devices = score_data[:, 1]
    scores = score_data[:, 0].astype(np.float)
    keys = score_data[:, 1]

    eer_cm = compute_eer17(scores, keys)
    return eer_cm


def compute_eer17(scores, keys):
    score_target = scores[keys == 'target']
    score_nontarget = scores[keys == 'nontarget']
    eer_cm = em.compute_eer(score_target, score_nontarget)[0]
    print('(genuine:{} + spoof:{}) EER \t= {:8.5f} % '.format(len(score_target), len(score_nontarget), eer_cm * 100))

    return eer_cm


if __name__ == '__main__':
    compute_eer17_from_file('dev-eer')
    compute_eer17_from_file('eer-file')
