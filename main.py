import argparse
# import bob.learn.em as em
# from bob.io.base import HDF5File
from scripts import data, classify, evaluate

parser = argparse.ArgumentParser(description='Baseline Code-switching Classifier')

parser.add_argument('--extract', action='store_true', default=False, help='enable feature extraction')
parser.add_argument('--train', action='store_true', default=False, help='enable train')
parser.add_argument('--test', action='store_true', default=False, help='enable test')

parser.add_argument('--window', type=int, default=64, help='Window size in ms')
parser.add_argument('--num_features', type=int, default=13, help='Number of MFCC features')
parser.add_argument('--num_components', type=int, default=64, help='Number of Gaussian components')

parser.add_argument('--save_path', type=str, default="models/default/", help='Path to dump models to')
parser.add_argument('--load_path', type=str, help='Path to load model from, must end in .joblib')

args = parser.parse_args()

if __name__ == '__main__':
    if args.extract:
        data.extract_with_test_utterance(args.window*16, args.num_features)
    if args.train:
        switch_data, non_switch_data = data.load(args.window, args.num_features)
        ubm = classify.fit_ubm(switch_data, non_switch_data, args.num_components, args.save_path)
        switch_gmm = classify.fit_adap(switch_data, non_switch_data, ubm, args.num_components, args.save_path)
    if args.test:
        switch_data, non_switch_data = data.load(args.window, args.num_features, test=True)
        if not args.train:
            try:
                switch_gmm = em.GMMMachine(HDF5File(args.load_path + 'switch.h5'))
                non_switch_gmm = em.GMMMachine(HDF5File(args.load_path + 'non_switch.h5'))
            except :
                print("Models not found."); exit()
        scores = classify.predict(switch_data, non_switch_data, switch_gmm, non_switch_gmm)
        Y,Y_pred = evaluate.get_predictions(scores)
        evaluate.evaluate(Y, Y_pred)
        # evaluate.confusion_matrix(Y, Y_pred)