import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Create npy files - train_non_switch.npy, train_switch.npy, test_non_switch.npy, test_switch.npy
# try 75 ms, 100 ms, 150 ms windows
# TODO: Read through text.bw
# For each utterance decide where switch points are using word_align
# Figure out where file is located and test or train
# extract MFCC features
# normalize
# put into correct numpy arrays
# write numpy arrays to disk

def get_file_locations():
    audio_locations_train = open('data/wav_train.scp')
    audio_location_lines_train = audio_locations_train.readlines()
    audio_locations_train.close()

    locations_train = {}
    print 'Loading train file locations...'
    for line in tqdm(audio_location_lines_train):
        line_data = line.split()
        if len(line_data) > 2:
            locations_train[line_data[0]] = line_data[2]

    audio_locations_test = open('data/wav_test.scp')
    audio_location_lines_test = audio_locations_test.readlines()
    audio_locations_test.close()

    locations_test = {}
    print 'Loading test file locations...'
    for line in tqdm(audio_location_lines_test):
        line_data = line.split()
        if len(line_data) > 2:
            locations_test[line_data[0]] = line_data[2]

    return (locations_train, locations_test)

def get_word_alignments():
    word_alignment_file = open('data/gale_word_align.ctm')
    word_alignment_lines = word_alignment_file.readlines()
    word_alignment_file.close()

    word_alignments = {}
    print 'Loading word alignments...'
    for line in tqdm(word_alignment_lines):
        line_data = line.split()
        utterance_list = line_data[0].split('_')[:-1]
        utterance_list.append(utterance_list.pop()[:-3])
        utterance = '_'.join(utterance_list)
        alignment_data = (line_data[4], (float(line_data[2]), float(line_data[3])))
        if utterance in word_alignments:
            word_alignments[utterance].append(alignment_data)
        else:
            word_alignments[utterance] = [alignment_data]

    return word_alignments

def cmvn_slide(X, win_len=300, cmvn=False):
    max_length = np.shape(X)[0]
    new_feat = np.empty_like(X)
    cur = 1
    left_win = 0
    right_win = int(win_len/2)

    for cur in range(max_length):
        cur_slide = X[cur-left_win:cur+right_win,:]
        mean = np.mean(cur_slide,axis=0)
        std = np.std(cur_slide,axis=0)
        if cmvn == 'mv':
            new_feat[cur,:] = (X[cur,:]-mean)/std # for cmvn
        elif cmvn == 'm':
            new_feat[cur,:] = (X[cur,:]-mean) # for cmn
        if left_win < win_len/2:
            left_win += 1
        elif max_length-cur < win_len/2:
            right_win -= 1
    return new_feat


def extract(window, num_features):
    locations_train, locations_test = get_file_locations()
    word_alignments = get_word_alignments()

    transcription_file = open('data/text.bw')
    transcription_lines = transcription_file.readlines()
    transcription_file.close()

    print 'Generating features...'
    for line in tqdm(transcription_lines):
        line_data = line.split()
        utterance_data = line_data[0]
        utterance_words = line_data[1:]

        utterance_data_list = utterance_data.split('_')

        file = '_'.join(utterance_data_list[:-2])
        alignment_identifier = utterance_data.split('.')[0]
        start = float(utterance_data_list[-2])
        stop = float(utterance_data_list[-1])

        alignment_data = word_alignments[alignment_identifier]

        if file in locations_train:
            test_or_train = 'train'
            file_location = locations_train[file]
        else:
            test_or_train = 'test'
            file_location = locations_test[file]

        file_location = 'data/ABUDHABI_ABUDHNEWS_ARB_20070206_115800.flac'
        y, sr = sf.read(file_location, start=int(16000*start), stop=int(16000*stop)+1)
        # each column represents 0.01 second
        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=num_features, n_fft=window, hop_length=160, fmin=133, fmax=6955)
        width = mfcc.shape[0]
        if width % 2 == 0:
            width -= 1
        mfcc_delta = librosa.feature.delta(mfcc, width=width)
        mfcc_delta_delta = librosa.feature.delta(mfcc, width=width, order=2)
        Y = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta))
        Y = cmvn_slide(Y, cmvn='m')

        record_switch = False
        switch_idxs = []
        for i in range(len(utterance_words)):
            word = utterance_words[i]
            if (word == '<non-MSA>' and i != 0) or word == '</non-MSA>':
                record_switch = True
            elif word == '<non-MSA>' and i == 0:
                continue
            else:
                word_data = alignment_data.pop(0)
                if record_switch:
                    # Add features for switch point
                    start = word_data[1][0]
                    idx = int(round(start*100.0))
                    switch_idxs.append(idx)
                    # Add features immediately surrounding (+/- 20 ms) switch point
                    if idx - 2 > -1:
                        switch_idxs.append(idx-2)
                    if idx - 1 > -1:
                        switch_idxs.append(idx-1)
                    if idx + 1 < Y.shape[1]:
                        switch_idxs.append(idx+1)
                    if idx + 2 < Y.shape[1]:
                        switch_idxs.append(idx+2)
                    record_switch = False

        print switch_idxs
        break
