import bob.learn.em as em
import numpy as np
from bob.io.base import HDF5File

def fit_ubm(switch_data, non_switch_data, num_components, save_path):
    print('Generating UBM...')
    all_data = np.concatenate((switch_data, non_switch_data))
    ubm = em.GMMMachine(num_components, all_data.shape[1])
    ubm_trainer = em.ML_GMMTrainer(True, True, True)
    em.train(ubm_trainer, ubm, all_data, max_iterations=12,
                convergence_threshold=0.001)
    ubm.save(HDF5File(save_path + 'ubm.h5', 'w'))
    return ubm

def fit_adap(switch_data, non_switch_data, ubm, num_components, save_path):
    print('Generating GMMs...')
    switch_gmm = em.GMMMachine(num_components, switch_data.shape[1])
    non_switch_gmm = em.GMMMachine(num_components, switch_data.shape[1])
    switch_trainer = em.MAP_GMMTrainer(ubm, relevance_factor=10, update_variances=False, update_weights=False)
    non_switch_trainer = em.MAP_GMMTrainer(ubm, relevance_factor=10, update_variances=False, update_weights=False)
    em.train(switch_trainer, switch_gmm, switch_data, max_iterations=14,
                convergence_threshold=0.001)
    em.train(non_switch_trainer, non_switch_gmm, non_switch_data, max_iterations=14,
                convergence_threshold=0.001)
    switch_gmm.save(HDF5File(save_path + 'switch.h5', 'w'))
    non_switch_gmm.save(HDF5File(save_path + 'non_switch.h5', 'w'))
    return (switch_gmm, non_switch_gmm)

def predict(switch_data, non_switch_data, switch_gmm, non_switch_gmm):
    return {'switch': list(map(lambda x: (switch_gmm.log_likelihood(x), non_switch_gmm.log_likelihood(x)), switch_data)),
            'non_switch': list(map(lambda x: (switch_gmm.log_likelihood(x), non_switch_gmm.log_likelihood(x)), non_switch_data))}