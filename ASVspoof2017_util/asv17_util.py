import h5py
import numpy as np
import pandas as pd

from ASVspoof2017_util import eval_metrics as em

label_encoder = {'genuine': 0, 'spoof': 1}


def read_protocol(protocol_file):
    protocol = pd.read_csv(protocol_file, header=None, sep=r' ', dtype=str, engine='python',
                           names=['file_id', 'type_id', 'speaker_id', 'phrase_id', 'environment_id',
                                  'playback_device_id', 'recording_device_id'])
    file_id = protocol['file_id']
    type_id = protocol['type_id']
    speaker_id = protocol['speaker_id']
    phrase_id = protocol['phrase_id']
    environment_id = protocol['environment_id']
    playback_device_id = protocol['playback_device_id']
    recording_device_id = protocol['recording_device_id']

    for i in range(len(file_id)):
        file_id[i] = file_id[i].replace('.wav', '')

    return file_id.values, type_id.values, speaker_id.values, phrase_id.values, environment_id.values, playback_device_id.values, recording_device_id.values



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
