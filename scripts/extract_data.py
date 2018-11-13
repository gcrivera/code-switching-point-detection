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
        alignment_data = (line_data[4], (line_data[2], line_data[3]))
        if utterance in word_alignments:
            word_alignments[utterance].append(alignment_data)
        else:
            word_alignments[utterance] = [alignment_data]

    return word_alignments


def extract(window, num_features):
    locations_train, locations_test = get_file_locations()
    word_alignments = get_word_alignments()

