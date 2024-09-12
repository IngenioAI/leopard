from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia
from tensorflow_privacy.privacy.privacy_tests import utils
from tensorflow.keras.utils import to_categorical
from typing import Optional
import tensorflow as tf
import numpy as np

from privacy_meter.dataset import Dataset

from target.torch_target import torch_predict
from attacks.config import priv_meter as pm


def get_attack_inp(model, tdata, is_torch):
    print("Collecting prediction stats....")

    if not is_torch:
        logits_train = model.predict(tdata.train_data)
        logits_test = model.predict(tdata.test_data)
    else:
        logits_train = torch_predict(
            model, tdata.train_data)
        logits_test = torch_predict(model, tdata.test_data)

    print('Apply softmax to get probabilities from logits...')
    prob_train = tf.nn.softmax(logits_train, axis=-1)
    prob_test = tf.nn.softmax(logits_test)
    print("prob_train", prob_train.shape)
    print('Compute losses...')
    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant
    print('train_labels', tdata.train_labels.shape)
    y_train_onehot = to_categorical(tdata.train_labels)
    y_test_onehot = to_categorical(tdata.test_labels)

    loss_train = cce(constant(y_train_onehot), constant(
        prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test_onehot), constant(
        prob_test), from_logits=False).numpy()
    # print('loss_train', loss_train.shape)
    # print('logits_train', logits_train.shape)
    # exit(0)
    adata = AttackInputData(
        logits_train=logits_train,
        logits_test=logits_test,
        loss_train=loss_train,
        loss_test=loss_test,
        labels_train=tdata.train_labels,
        labels_test=tdata.test_labels
    )
    return adata


def get_stat_and_loss_aug(model,
                          x,
                          y,
                          sample_weight: Optional[np.ndarray] = None,
                          batch_size=64,
                          is_torch=False):

    losses, stat = [], []
    print('Computing stats from loss and logits....')

    if is_torch:
        logits = torch_predict(model, x)
    else:
        logits = model.predict(x, batch_size=batch_size)
    prob = amia.convert_logit_to_prob(logits)

    losses.append(utils.log_loss(
        y, prob, sample_weight=sample_weight))

    stat.append(
        amia.calculate_statistic(
            prob, y, sample_weight=sample_weight, is_logits=False))

    return np.vstack(stat.copy()).transpose(1, 0), np.vstack(losses.copy()).transpose(1, 0)


def get_trg_ref_data(tdata, num_class, population=False):
    x_train_all = tdata.train_data
    x_test_all = tdata.test_data

    if not isinstance(tdata.train_data, np.ndarray):
        x_train_all = np.concatenate(
            [data for data, _ in tdata.train_data], axis=0)
        x_test_all = np.concatenate(
            [data for data, _ in tdata.test_data], axis=0)

    y_train_all = tf.keras.utils.to_categorical(
        tdata.train_labels, num_class)
    y_test_all = tf.keras.utils.to_categorical(
        tdata.test_labels, num_class)

    x_train, y_train = x_train_all[:pm['num_train_points']
                                   ], y_train_all[:pm['num_train_points']]
    x_test, y_test = x_test_all[:pm['num_test_points']
                                ], y_test_all[:pm['num_test_points']]

    train_ds = {'x': x_train, 'y': y_train}
    test_ds = {'x': x_test, 'y': y_test}

    target_dataset = Dataset(
        data_dict={'train': train_ds, 'test': test_ds},
        default_input='x', default_output='y'
    )

    if not population:
        return target_dataset

    else:
        x_population = x_train_all[pm['num_train_points']:(
            pm['num_train_points'] + pm['num_population_points'])]
        y_population = y_train_all[pm['num_train_points']:(
            pm['num_train_points'] + pm['num_population_points'])]

        population_ds = {'x': x_population, 'y': y_population}
        reference_dataset = Dataset(
            data_dict={'train': population_ds},
            default_input='x', default_output='y'
        )
        return target_dataset, reference_dataset


def plot_curve_with_area(x, y, xlabel, ylabel, ax, label, title=None):
    ax.plot([0, 1], [0, 1], 'k-', lw=1.0)
    ax.plot(x, y, lw=2, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set(aspect=1, xscale='log', yscale='log')
    ax.title.set_text(title)

def convert_result_list(audit_result_list):
    result_list = []
    for audit_result in audit_result_list:
        result = dict()
        item_key = ['metric_id', 'roc_auc']
        list_key = ['accuracy','tn', 'tp', 'fp', 'fn']
        for item in item_key:
            result[item] = audit_result.__dict__[item]
        for item in list_key:
            if type(audit_result.__dict__[item]) == np.ndarray:
                result[item] = audit_result.__dict__[item].tolist()
            elif type(audit_result.__dict__[item]) == np.int64:
                result[item] = int(audit_result.__dict__[item])
            else:
                result[item] = audit_result.__dict__[item]
        result_list.append(result)
    return result_list