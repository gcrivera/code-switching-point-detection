from tqdm import tqdm
import numpy as np

def split_data():
    train_13_non = np.load('data/train_non_switch_w_64_f_13.npy')
    train_13_switch = np.load('data/train_switch_w_64_f_13.npy')
    train_20_non = np.load('data/train_non_switch_w_64_f_20.npy')
    train_20_switch = np.load('data/train_switch_w_64_f_20.npy')
    np.random.shuffle(train_13_non)
    np.random.shuffle(train_13_switch)
    np.random.shuffle(train_20_non)
    np.random.shuffle(train_20_switch)

    total_non = train_13_non.shape[0]
    total_switch = train_13_switch.shape[0]

    train_non_idx = int(total_non * 0.7)
    train_switch_idx = int(total_switch * 0.7)

    np.save('data/train_non_switch_w_64_f_13_temp.npy', train_13_non[:train_non_idx])
    np.save('data/test_non_switch_w_64_f_13_temp.npy', train_13_non[train_non_idx:])

    np.save('data/train_switch_w_64_f_13_temp.npy', train_13_switch[:train_switch_idx])
    np.save('data/test_switch_w_64_f_13_temp.npy', train_13_switch[train_switch_idx:])

    np.save('data/train_non_switch_w_64_f_20_temp.npy', train_20_non[:train_non_idx])
    np.save('data/test_non_switch_w_64_f_20_temp.npy', train_20_non[train_non_idx:])

    np.save('data/train_switch_w_64_f_20_temp.npy', train_20_switch[:train_switch_idx])
    np.save('data/test_switch_w_64_f_20_temp.npy', train_20_switch[train_switch_idx:])

if __name__ == '__main__':
    split_data()
