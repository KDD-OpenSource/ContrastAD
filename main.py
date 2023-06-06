import json
import os
import itertools
import pickle
from pathlib import Path
import shutil

from model.experiment import Experiment_runner
from model.evaluation import run_evaluation, latent_plot
from model.datasets import MSL, ECG5000, SMAP, UCR, SWaT
from model.competitors.dagmm import DAGMMPredictor
from model.competitors.iforest import iForest
from model.competitors.lof import Lof
from model.competitors.lstm_autoencoder import LSTMAE
from model.competitors.ocsvm import OCSVM

root = Path(__file__).resolve().parent
data_root = os.path.join(root, 'dataset')

run_id = 'experiment'
output_root = os.path.join(root, 'results', run_id)

with open(f'{root}/configuration/config.json', 'r') as f:
    config = json.load(f)
datasets = []
datasets += [f'ucr_{n:03d}' for n in range(1, 6)]
datasets += [f'msl_{channel}' for channel in ['M-6', 'M-1', 'M-2', 'S-2', 'P-10']]
datasets += [f'smap_{channel}' for channel in ['P-1', 'S-1', 'E-1', 'E-2', 'E-3']]
datasets += ['swat', 'ecg_5000']
gpu = 0
verbose = False

dataset_obj = {'ecg_5000': ECG5000, 'smap': SMAP, 'ucr': UCR, 'msl': MSL, 'swat': SWaT}
competitors = [iForest, Lof, OCSVM, LSTMAE, DAGMMPredictor]


def train_pred_func(train_set, test_set, win_label, experiment_dir, exp_config, gpu_name, true_anomaly_batch):
    print("*" * 25)
    print("*" * 5, "Model: ContrastAD", "*" * 5)
    print("*" * 25)
    runner = Experiment_runner(experiment_dir, exp_config, gpu_name, verbose=verbose)
    model = runner.train(train_set, true_anomaly_batch)

    if model is None:
        print("No sufficient data window under current parameter setting, batch size= ", exp_config['batch_size'])
        return 'NoData'

    if os.path.exists(f'{experiment_dir}/eval.pkl'):
        print("Prediction result exists in the output directory. Skipping prediction.")
        return True

    anomaly_scores, valid_label, augmented_embeddings_collector, details = runner.predict(test_set, model, win_label,
                                                                                          true_anomaly_batch)
    if valid_label.sum() == 0:
        print("No anomaly in test set under current parameter setting, batch size= ", exp_config['batch_size'])
        return 'NoAno'

    pred, eval_dict = run_evaluation('contrastAD', None, anomaly_scores, valid_label)
    if verbose:
        latent_plot(augmented_embeddings_collector['X_embedding'],
                    augmented_embeddings_collector['negative_augmentation_embeddings'],
                    augmented_embeddings_collector['positive_augmentation_embeddings'],
                    valid_label,
                    experiment_dir)
    with open(f'{experiment_dir}/details.pkl', 'wb') as fp:
        pickle.dump(details, fp)
    with open(f'{experiment_dir}/eval.pkl', 'wb') as fp:
        pickle.dump(eval_dict, fp)
    return True


def run_competitors(train_set, test_set, win_label, experiment_dir, window_length):
    for competitor in competitors:
        competitor_model = competitor(window_length, gpu=gpu)
        if os.path.exists(f'{experiment_dir}/eval_{competitor_model.name}.pkl'):
            print("Competitor result exists in the output directory. Skipping prediction.")
            continue
        print("*" * 25)
        print("*" * 5, "Model: ", competitor_model.name, "*" * 5)
        print("*" * 25)
        competitor_model.train(train_set)
        window_pred, window_scores, real_window_label = competitor_model.test(test_set, win_label)
        pred, eval_dict = run_evaluation(competitor_model.name, window_pred, window_scores, real_window_label)

        with open(f'{experiment_dir}/eval_{competitor_model.name}.pkl', 'wb') as fp:
            pickle.dump(eval_dict, fp)


def main():
    for dataset in datasets:
        print("Dataset: ", dataset)
        if dataset.startswith('ucr'):
            subset = dataset.split('_')[-1]
            dataset = 'ucr'
        elif dataset.startswith('msl'):
            subset = dataset.split('_')[-1]
            dataset = 'msl'
        elif dataset.startswith('smap'):
            subset = dataset.split('_')[-1]
            dataset = 'smap'
        elif dataset.startswith('smd'):
            subset = dataset.split('_')[-1]
            dataset = 'smd'
        else:
            subset = ''
        exp_config = {**config[dataset], **config['model']}
        for (input_size, window_size, hidden_dim, num_layers, dropout_rate, direction, encoder, lr, batch_size, epochs,
             tcn_kernel_size, val_percentage, temperature, latent_augmentation, seed, spike_scope, spike_lambda,
             spike_locality, noise_lambda, reset_eps, scale_lambda, transformation_mode, positive_transformation,
             negative_transformation) in \
                itertools.product(*exp_config.values()):
            x_train, test_anomalies, x_test, y_test, win_label = dataset_obj[dataset](win_len=window_size,
                                                                                      subset=subset).data()
            input_size = x_train.shape[1]  # do not remove
            true_anomaly_batch = None
            single_config = dict()
            for param in exp_config.keys():
                single_config[param] = eval(param)

            print(single_config)
            identifier = '_'.join([str(x) for x in single_config.values()])
            suffix = '_' + subset if subset != '' else ''
            output_dir = os.path.join(output_root, dataset + suffix, identifier)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            success = train_pred_func(x_train, x_test, win_label, output_dir, single_config, gpu, true_anomaly_batch)

            if success == 'NoData':
                os.rmdir(output_dir)
            elif success == 'NoAno':
                shutil.rmtree(output_dir)
            else:
                with open(f'{output_dir}/config.pkl', 'wb') as f:
                    pickle.dump(single_config, f)
        competitor_output_dir = os.path.join(output_root, dataset + suffix)
        run_competitors(x_train, x_test, win_label, competitor_output_dir, window_size)


if __name__ == "__main__":
    main()
