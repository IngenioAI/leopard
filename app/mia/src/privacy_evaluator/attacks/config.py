import tensorflow as tf
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

aconf = {
    'lr': 0.01,
    'batch_size': 64,
    'epochs': 10,
    'n_shadows': 2,
    'shpath': './attacks/shadows'
}

priv_meter = {
    'num_train_points': 10000,
    'num_test_points': 10000,
    'epochs': 10,
    'batch_size': 64,
    'num_population_points': 10000,
    'fpr_tolerance_list': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'input_shape': (224, 224, 3),
    'ref_models': './attacks/shadows/',
    'n_shadows': 2,
    'torch_loss': torch.nn.CrossEntropyLoss(reduction='none'),
    'tf_loss': tf.keras.losses.CategoricalCrossentropy()
}
