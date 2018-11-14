from tqdm import tqdm

def split_data():
    transcription = open('data/text.bw')
    transcription_lines = transcription.readlines()
    transcription.close()

    audio_locations_train = open('data/wav_train.scp')
    audio_location_lines_train = audio_locations_train.readlines()

    locations_train = {}
    print 'Loading train file locations...'
    for line in tqdm(audio_location_lines_train):
        line_data = line.split()
        if len(line_data) > 2:
            locations_train[line_data[0]] = line_data[2]

    audio_locations_test = open('data/wav_test.scp')
    audio_location_lines_test = audio_locations_test.readlines()

    locations_test = {}
    print 'Loading test file locations...'
    for line in tqdm(audio_location_lines_test):
        line_data = line.split()
        if len(line_data) > 2:
            locations_test[line_data[0]] = line_data[2]

    total_missing = 0
    for line in transcription_lines:
        line_data = line.split(' ')
        filename = ''.join(map(lambda x: x + '_', line_data[0].split('_')[:-2]))[:-1]
        if filename not in locations_train.keys() and filename not in locations_test.keys():
            total_missing += 1

    print 'Total missing from wav_*.scp ' + str(total_missing)

