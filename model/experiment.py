import os
import time
import random
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from .transformations import Transformation
from .contrastive_loss import contrastive_loss_func_new
from .contrastAD import ContrastADModel


def device(gpu):
    if torch.cuda.is_available():
        d = f'cuda:{gpu}'
    else:
        d = 'cpu'
    return d


class Experiment_runner:
    def __init__(self, experiment_dir, config, gpu, verbose=False):
        self.positive_transformation = config['positive_transformation'].split(';')
        self.negative_transformation = config['negative_transformation'].split(';')
        self.config = config
        self.experiment_dir = experiment_dir
        self.gpu = gpu
        self.lr = config['lr']
        self.window_size = config['window_size']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.val_percentage = config['val_percentage']
        self.temperature = config['temperature']
        self.latent_augmentation = config['latent_augmentation']
        self.seed = config['seed']
        self.transform_handler = Transformation(self.seed)
        self.spike_scope = config['spike_scope']
        self.spike_lambda = config['spike_lambda']
        self.spike_locality = config['spike_locality']
        self.noise_lambda = config['noise_lambda']
        self.reset_eps = config['reset_eps']
        self.scale_lambda = config['scale_lambda']
        self.transformation_mode = config['transformation_mode']  # random, all
        self.verbose = verbose

    def _load_model(self, existing_model=None):
        model = ContrastADModel(self.config)
        if existing_model is not None:
            model.load_state_dict(torch.load(existing_model))
            model.eval()
        model = self._to_device(model)
        return model

    def _save_checkpoint(self, model):
        model_output_path = os.path.join(self.experiment_dir, "checkpoint.th")
        torch.save(model.state_dict(), model_output_path)

    def _to_device(self, obj):
        obj = obj.to(device(self.gpu))
        return obj

    def _augmentation(self, batch, true_anomaly_batch):
        negative_augmentation = self._transformation(batch, self.negative_transformation, true_anomaly_batch)
        positive_augmentation = self._transformation(batch, self.positive_transformation, true_anomaly_batch)

        return negative_augmentation, positive_augmentation

    def loss_plot(self, train_loss, val_loss):
        ax = pd.Series(train_loss).plot(c='blue', label='Train')
        pd.Series(val_loss).plot(ax=ax, c='orange')
        ax.legend(['Train', 'Val'])
        fig = ax.get_figure()
        fig.savefig(f'{self.experiment_dir}/loss.png')

        fig.clear()

    def _transformation(self, batch, transformations, true_anomaly_batch=None):

        if self.transformation_mode == 'random':
            random.seed(self.seed)
            idx = random.randint(0, len(transformations) - 1)
            transformations = [transformations[idx]]
        transformed_batchs = []
        for transformation in transformations:
            new_batch = []
            for window in batch:
                transformed = window.clone()
                if 'spike' == transformation:
                    transformed = self.transform_handler.add_spike(transformed, self.spike_scope, self.spike_lambda,
                                                                   self.spike_locality)
                if 'shuffle' == transformation:
                    transformed = self.transform_handler.shuffle(transformed)
                if 'reset' == transformation:
                    transformed = self.transform_handler.reset(transformed, self.reset_eps)
                if 'trend' == transformation:
                    transformed = self.transform_handler.trend(transformed)
                if 'repeat' == transformation:
                    transformed = self.transform_handler.repeat(transformed)
                if 'add_noise' == transformation:
                    transformed = self.transform_handler.add_noise(transformed, self.noise_lambda)
                if 'scale' == transformation:
                    transformed = self.transform_handler.scale(transformed, self.scale_lambda)
                if 'true_anomaly' == transformation:
                    transformed = true_anomaly_batch.clone().float()
                    return [self._to_device(transformed)]
                new_batch.append(transformed)
            transformed_batchs.append(new_batch)
        return [torch.stack(b) for b in transformed_batchs]

    def train(self, train_set, true_anomaly_batch=None, existing_model=None):
        sequences = [train_set[i:i + self.window_size] for i in
                     range(0, train_set.shape[0] - self.window_size + 1, self.window_size)]

        np.random.seed(self.seed)
        indices = np.random.permutation(len(sequences))
        split_point = int(self.val_percentage * len(sequences))

        train_data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                       sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        validation_data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                            sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)
        if len(train_data_loader) == 0:
            print("Train loader size: ", len(train_data_loader))
            return None

        if os.path.exists(f'{self.experiment_dir}/training_details.pkl'):
            print('Pretrained model exists in the output directory.')
            model = self._load_model(existing_model=os.path.join(self.experiment_dir, "checkpoint.th"))
            return model

        model = self._load_model(existing_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            print("Using CUDA device")

        print("Beginning training.")

        training_loss = []
        validation_loss = []

        for epoch in range(1, self.epochs + 1):

            start_time = time.time()

            print(f'Epoch {epoch}/{self.epochs - 1}')

            model.train()
            running_train_loss = 0.0

            for i, batch in enumerate(tqdm(iter(train_data_loader))):
                optimizer.zero_grad()
                negative_augmentation, positive_augmentation = self._augmentation(batch.float(), true_anomaly_batch)

                X_embeddings = model.forward(self._to_device(batch.float()))
                _negative_augmentation_embeddings = [model.forward(self._to_device(na)) for na in negative_augmentation]
                _positive_augmentation_embeddings = [model.forward(self._to_device(pa)) for pa in positive_augmentation]

                # negative_augmentation_embeddings = torch.concat(_negative_augmentation_embeddings, axis=0)
                # positive_augmentation_embeddings = torch.concat(_positive_augmentation_embeddings, axis=0)
                loss = contrastive_loss_func_new(X_embeddings,
                                                 _negative_augmentation_embeddings,
                                                 _positive_augmentation_embeddings,
                                                 temperature=self.temperature,
                                                 seed=self.seed,
                                                 latent_augmentation=self.latent_augmentation)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            epoch_training_loss = running_train_loss / (i + 1)
            training_loss.append(epoch_training_loss)

            model.eval()
            running_val_loss = 0.0

            for i, batch in enumerate(validation_data_loader):
                negative_augmentation, positive_augmentation = self._augmentation(batch.float(), true_anomaly_batch)
                X_embeddings = model.forward(self._to_device(batch.float()))
                _negative_augmentation_embeddings = [model.forward(self._to_device(na)) for na in
                                                     negative_augmentation]
                _positive_augmentation_embeddings = [model.forward(self._to_device(pa)) for pa in
                                                     positive_augmentation]

                # negative_augmentation_embeddings = torch.concat(_negative_augmentation_embeddings, axis=0)
                # positive_augmentation_embeddings = torch.concat(_positive_augmentation_embeddings, axis=0)
                #
                val_loss = contrastive_loss_func_new(X_embeddings,
                                                     _negative_augmentation_embeddings,
                                                     _positive_augmentation_embeddings,
                                                     temperature=self.temperature,
                                                     seed=self.seed,
                                                     latent_augmentation=self.latent_augmentation)
                running_val_loss += val_loss.item()

            epoch_val_loss = running_val_loss / (i + 1)
            validation_loss.append(epoch_val_loss)

            self._save_checkpoint(model)

            used_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch, self.epochs, epoch_training_loss, epoch_val_loss, used_time))

            training_details = {'train_loss': training_loss, 'val_loss': validation_loss, 'time': used_time}

            with open(f'{self.experiment_dir}/training_details.pkl', 'wb') as f:
                pickle.dump(training_details, f)

            self.loss_plot(training_loss, validation_loss)

        return model

    def predict(self, test_set, existing_model, win_label, true_anomaly_batch=None):

        model = existing_model

        if torch.cuda.is_available():
            print("Using CUDA device")

        print("Beginning predicting.")

        sequences = [test_set[i:i + self.window_size] for i in
                     range(0, test_set.shape[0] - self.window_size + 1, self.window_size)]

        test_data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True, pin_memory=True)
        anomaly_score_collector = []
        augmentation_collector = []
        augmented_embeddings_collector = {'X_embedding': [],
                                          'negative_augmentation_embeddings': [],
                                          'positive_augmentation_embeddings': []}
        start_time = time.time()
        for i, batch in enumerate(tqdm(iter(test_data_loader))):
            negative_augmentation, positive_augmentation = self._augmentation(batch.float(), true_anomaly_batch)
            X_embeddings = model.forward(self._to_device(batch.float()))
            _negative_augmentation_embeddings = [model.forward(self._to_device(na)) for na in negative_augmentation]
            _positive_augmentation_embeddings = [model.forward(self._to_device(pa)) for pa in positive_augmentation]

            negative_augmentation_embeddings = torch.concat(_negative_augmentation_embeddings, axis=0) if len(
                _negative_augmentation_embeddings) != 0 else None
            positive_augmentation_embeddings = torch.concat(_positive_augmentation_embeddings, axis=0) if len(
                _positive_augmentation_embeddings) != 0 else None

            augmented_embeddings_collector['X_embedding'].append(X_embeddings.detach().cpu().numpy())
            if negative_augmentation_embeddings is not None:
                augmented_embeddings_collector['negative_augmentation_embeddings'].append(
                    negative_augmentation_embeddings.detach().cpu().numpy())
            if positive_augmentation_embeddings is not None:
                augmented_embeddings_collector['positive_augmentation_embeddings'].append(
                    positive_augmentation_embeddings.detach().cpu().numpy())

            augmentation_collector.append({'batch:': batch.float(),
                                           'negative': negative_augmentation,
                                           'positive': positive_augmentation})

            anomaly_score = contrastive_loss_func_new(X_embeddings,
                                                      _negative_augmentation_embeddings,
                                                      _positive_augmentation_embeddings,
                                                      temperature=self.temperature,
                                                      seed=self.seed,
                                                      latent_augmentation=self.latent_augmentation,
                                                      type='anomaly_score')
            anomaly_score_collector.append(anomaly_score.detach().cpu())
        anomaly_scores = torch.stack(anomaly_score_collector).ravel().numpy()

        used_time = time.time() - start_time
        valid_label = win_label[:anomaly_scores.shape[0]]

        for embedding_type in augmented_embeddings_collector.keys():
            if len(augmented_embeddings_collector[embedding_type]) != 0:
                augmented_embeddings_collector[embedding_type] = np.concatenate(
                    augmented_embeddings_collector[embedding_type], axis=0)
            else:
                augmented_embeddings_collector[embedding_type] = None
        if self.verbose:
            details = {'anomaly_scores': anomaly_scores,
                       'labels': valid_label,
                       'augmentation_collector': augmentation_collector,
                       'augmented_embeddings_collector': augmented_embeddings_collector,
                       'time': used_time}
        else:
            details = dict()

        return pd.Series(anomaly_scores), pd.Series(valid_label.ravel()), augmented_embeddings_collector, details
