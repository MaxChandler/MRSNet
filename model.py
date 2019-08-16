#!/usr/bin/env python2.7
from __future__ import print_function

import os
import matplotlib
# for headless mode
if not "DISPLAY" in os.environ: matplotlib.use("Agg")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import csv
import dill
import argparse
import seaborn as sns
import tensorflow as tf
from pylab import *
from Utilities.utils import *
from networks import *
from Utilities.constants import FULL_METABOLITE_LIST, OMEGA
from Simulators.PyGamma.pygamma_simulator import pygamma_spectra
from Dataset import Dataset
from Basis import Basis
from tensorflow.python.keras.utils import plot_model
from shutil import copyfile
from scipy.stats import linregress


SAVE_ROOT = ''

def startup():
    global SAVE_ROOT
    SAVE_ROOT = './'
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
    arg_parse()


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', help='Mode: "quantify" or "train"')

    # train settings
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-d', '--data_source', type=str, default='lcmodel',
                        help='Datasource for the basis spectra: LCModel, FID-A (requires Matlab), PyGamma.')
    parser.add_argument('-N', '--n_spectra', type=int, default=5000,
                        help='Number of training spectra to generate for training the network. Validation dataset contains n_spectra *0.2 samples.')

    parser.add_argument('--model_name', type=str, default='mrsnet_small_kernel_no_pool', dest='model',
                        help='Model function name from networks.py, default is mrsnet_small_kernel_no_pool')
    parser.add_argument('--label-norm', type=str, dest='norm', default='sum', help='Label normalisation: sum or max')
    parser.add_argument('--scanner_manufacturer', type=str, dest='scanner', default='siemens',
                        help='Scanner manufacturer: Siemes, GE or Phillips')
    parser.add_argument('--linewidths', type=float, nargs='+', dest='linewidths', default=None,
                        help='Linewidths to be used for simulation, default is 1.')
    parser.add_argument('--omega', type=float, dest='omega', default=OMEGA,
                        help='Scanner frequency in Hz, defaults to 123.23 (2.98T)')
    parser.add_argument('--metabolites', type=str, nargs='+', dest='metabolites', default=FULL_METABOLITE_LIST,
                        help='List of metabolites to train the network on, or to quantify. '
                             'Please use full names as defined in Utilities.utils.convert_mol_names')

    # data export settings
    parser.add_argument('--datatype', type=str, nargs='+', dest='datatype', default=['magnitude'])
    parser.add_argument('--acquisitions', type=int, nargs='+', dest='acquisitions', default=[0, 2])

    # quantification settings
    parser.add_argument('--network_path', help='Path of the network to load to quantifiy spectra',
                        default='./models/complete/MRSNet LCModel/')
    parser.add_argument('--spectra_path', help='Directory of spectra to quantify. This will search recursively.',
                        default='./Datasets/Benchmark/E1/MEGA_Combi_WS_ON/')


    args = parser.parse_args()

    # verify the molecule names and convert them
    convert_mol_names(args.metabolites, mode='verify')
    args.metabolites = convert_mol_names(args.metabolites, mode='lengthen')

    # lowercase the datatypes
    args.datatype = [x.lower() for x in args.datatype]

    if args.mode == 'train':
        basis = load_basis(args.data_source, args.scanner, args.omega, args.linewidths, args.metabolites)
        train_network(basis,
                      model_name=args.model,
                      epochs=args.epochs,
                      n_train_spectra=args.n_spectra,
                      label_norm=args.norm,
                      batch_size=args.batch_size,
                      metabolites=args.metabolites,
                      export_acquisitions=args.acquisitions,
                      export_datatype=args.datatype)
    elif args.mode == 'quantify':
        quantify(args.spectra_path, args.network_path, args.metabolites)
    else:
        raise Exception('Unknown mode ' + args.mode + '. Pick from ["train","test"]')


def quantify(ima_dir, model_dir, metabolites=None):
    basis = Basis.load_dicom(ima_dir)
    model = load_network(model_dir)

    if metabolites:
        # check to see if the network can actually quantify the requested metabolites
        if not all([m_name.lower() in [x.lower() for x in model.output_labels] for m_name in metabolites]):
            raise Exception('Network is unable to quantify one or more metabolites suplied.'
                            'Network is able to quantify: ' + str(model.output_labels) + '\n'
                            'Requested metabolites: ' + str(metabolites))
    else:
        metabolites = model.output_labels

    # generate a dataset from the loaded dicoms
    dataset = Dataset()
    dataset._name = 'quantify'
    dataset.basis = basis
    dataset.copy_basis()
    dataset.export_datatype = model.export_datatype
    dataset.export_acquisitions = model.export_acquisitions
    dataset.conc_normalisation = model.output_normalisation
    dataset.export_nu = False
    dataset.export_dss_peak = False
    dataset.add_adc_noise = False
    dataset.add_nu_noise = False

    # export the dataset into the format we're looking
    t_data, t_labels, mo_labels = dataset.export_to_keras(model_labels=metabolites)
    t_data, input_shape = reshape_data(t_data)

    # run the model in test mode to display the loss and accuracy
    model.evaluate(x=t_data, y=t_labels)
    # quantify!
    predictions = model.predict(t_data)

    # trim the output data to match the metabolites of interest
    cols_index = [[x.lower() for x in model.output_labels].index(m_name.lower()) for m_name in metabolites]
    predictions = predictions[:, cols_index]
    # renormalise the output labels
    predictions = normalise_labels(predictions, model.output_normalisation)

    # get the spectra in the order they were exported in dataset.export_to_keras
    # so we can match the predictions to the spectra
    spectra = np.array(dataset.group_spectra_by_id())

    # print the results table!
    print('\n\nQuantifying %d MEGA-PRESS Spectra' % (len(spectra)))
    print('This network can only quantify: ' + str(model.output_labels))
    print('\tNetwork path: ' + model_dir)
    print('\tDICOM path: ' + ima_dir + '\n\n')

    for spec, prediction in zip(spectra, predictions):
        if sum(spec[0].concentrations):
            print('Spectra ID: ' + spec[0].id)
            print('\t%-*s %s %s' % (20, 'Metabolite', 'Predicded', 'Actual'))
            actual_concentrations = normalise_labels(spec[0].concentrations, model.output_normalisation)
            for p, a, m_name in zip(prediction, actual_concentrations, convert_mol_names(metabolites, mode='lengthen')):
                print('\t%-*s %.6f %.6f' % (20, m_name, p, a))
            print('\n')
        else:
            print('Spectra ID: ' + spec[0].id)
            print('\t%-*s %s' % (20, 'Metabolite', 'Predicted'))
            for p, m_name in zip(prediction, convert_mol_names(metabolites, mode='lengthen')):
                print('\t%-*s %.6f' % (20, m_name, p))
            print('\n')


def load_basis(basis_source, scanner_manufacturer, omega, linewidths, metabolites):
    # can return multiple basis for training when more than one linewidth is supplied
    basis_source = basis_source.lower()
    scanner_manufacturer = scanner_manufacturer.lower()

    if basis_source == 'lcmodel':
        if linewidths:
            raise Exception('Cannot supply LCModel basis set with linewidths argument. It is not a simulator option; it has one fixed linewidth.')
        if scanner_manufacturer == 'siemens':
            basis = [Basis.load_lcm_basis(metabolite_names=metabolites, megapress=True,
                                          edit_off='./Basis/simulated/LCModel/MEGAPRESS_edit_off_Siemens_3T.basis',
                                          difference='./Basis/simulated/LCModel/MEGAPRESS_difference_Siemens_3T_kasier.basis',
                                          omega=omega)]
        elif scanner_manufacturer == 'ge':
            basis = [Basis.load_lcm_basis(metabolite_names=metabolites, megapress=True,
                                          edit_off='./Basis/simulated/LCModel/MEGAPRESS_edit_off_GE_3T.basis',
                                          difference='./Basis/simulated/LCModel/MEGAPRESS_difference_GE_3T_kasier.basis',
                                          omega=omega)]
        elif scanner_manufacturer == 'phillips':
            basis = [Basis.load_lcm_basis(metabolite_names=metabolites, megapress=True,
                                          edit_off='./Basis/simulated/LCModel/MEGAPRESS_edit_off_Phillips_3T.basis',
                                          difference='./Basis/simulated/LCModel/MEGAPRESS_difference_Phillips_3T_kasier.basis',
                                          omega=omega)]
        else:
            raise Exception('No LCModel basis set found for scanner: ' + scanner_manufacturer)
    elif basis_source == 'fida':
        basis = []
        if scanner_manufacturer == 'siemens':
            if linewidths:
                for linewidth in linewidths:
                    basis.append(Basis.load_fida(metabolites, linewidth=linewidth, omega=omega))
            else:
                basis.append(Basis.load_fida(metabolites, linewidth=1, omega=omega))
        else:
            raise Exception('No FID-A simulator found for scanner: ' + scanner_manufacturer)

    elif basis_source == 'pygamma':
        basis = []
        if scanner_manufacturer == 'siemens':
            if linewidths:
                for linewidth in linewidths:
                    basis.append(pygamma_spectra(metabolites, pulse_sequence='megapress', linewidth=linewidth, omega=omega))
            else:
                basis.append(pygamma_spectra(metabolites, pulse_sequence='megapress', linewidth=1, omega=omega))
        else:
            raise Exception('No PyGamma simulator found for scanner: ' + scanner_manufacturer)

    else:
        raise Exception('Unknown basis source: ' + basis_source + '. Please choose from ["lcmocel","fida","pygamma"]')

    return basis


def conv_network(data, labels, output_labels, dataset_name, model_name, epochs, batch_size, label_norm='sum',
                 export_datatype=None,
                 export_acquisitions=None,
                 dataset_normalisation=None,
                 save_directory=SAVE_ROOT + 'models/'):

    global SAVE_ROOT

    _MODEL_NAME = datetime.datetime.now().strftime('%d%m%y_%H:%M:%S') + '_'
    _MODEL_NAME += model_name + '_'
    _MODEL_NAME += 'e_' + str(epochs) + '_'
    _MODEL_NAME += 'b_' + str(batch_size) + '_'
    _MODEL_NAME += 'n_' + str(len(data)) + '_'

    data, input_shape = reshape_data(data)
    _MODEL_NAME += 'shp_' + str(input_shape).replace('(', '').replace(')', '').replace(', ', '_')

    models = []
    n_classes = len(labels[0])  # don't remove me - used in the following eval statements

    # Been disabled as XLA_GPUs can appear multiple times in the num_gpus() method.
    # if num_gpus() > 1:
    #     # multi gpu support: https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus
    #     import tensorflow as tf
    #     with tf.device('/cpu:0'):
    #         models.append(eval(model_name + '(input_shape, n_classes)'))
    #     models.append(keras.utils.multi_gpu_model(models[0], gpus=num_gpus()))
    #     print('Model split over ' + str(num_gpus()) + ' GPUs')
    # elif num_gpus() == 1:
    #     with tf.device('/gpu:0'):
    #         models.append(eval(model_name + '(input_shape, n_classes)'))
    # else:
    #     with tf.device('/cpu:0'):
    #         models.append(eval(model_name + '(input_shape, n_classes)'))

    models.append(eval(model_name + '(input_shape, n_classes)'))

    if label_norm == 'sum':
        if models[0].layers[-1].activation.func_name != 'softmax':
            raise Exception('When using "sum" label norm, please use softmax activation for final layer.')
    elif label_norm == 'max':
        if models[0].layers[-1].activation.func_name != 'sigmoid':
            raise Exception('When using "max" label norm, please use sigmoid activation for final layer.')

    for sub_directory in ['complete', 'incomplete']:
        if not os.path.exists(os.path.join(save_directory, sub_directory)):
            os.makedirs(os.path.join(save_directory, sub_directory))

    save_path = save_directory + 'incomplete/' + _MODEL_NAME + '/'

    if os.path.isdir(save_path):
        raise Exception('There is already a model with this name')
    else:
        os.makedirs(save_path)

    plot_model(models[0],
               to_file=save_path + '/network_structure.png',
               show_shapes=True,
               show_layer_names=True)

    plot_label_distribution(labels, save_path, output_labels)

    optimiser = keras.optimizers.Adam(lr=1e-4,
                                      beta_1=0.9,
                                      beta_2=0.999,
                                      amsgrad=False)

    for model in models:
        model.compile(loss='mse',
                      optimizer=optimiser,
                      metrics=['acc', 'mse', 'mae'])

        model._MODEL_NAME = _MODEL_NAME
        model.output_labels = output_labels
        model.output_normalisation = dataset_normalisation

    model_metadata = {'_MODEL_NAME': _MODEL_NAME,
                      'output_labels': output_labels,
                      'output_normalisation': label_norm,
                      'dataset_name': dataset_name,
                      'export_datatype': export_datatype,
                      'export_acquisitions': export_acquisitions}

    models[-1].summary()

    callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=1e-12,
                                               patience=15,
                                               verbose=1,
                                               restore_best_weights=True)]

    history = models[-1].fit(x=data,
                             y=labels,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1,
                             shuffle=True,
                             callbacks=callbacks)

    save_network(save_path, models[0], model_metadata)

    score = models[-1].evaluate(data, labels, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    history = save_history(dataset_name, save_path, history.history, prefix='training_history', last_iteration=True)
    plot_history(history, save_path, prefix='train_')

    # alter the save path to be more descriptive
    old_save_path = save_path
    save_path = save_path.rstrip('/')

    # include the final later activaiton for reference
    save_path += '_actvn_' + models[-1].layers[-1].activation.func_name

    # and the dataset label norm technique
    if dataset_normalisation:
        save_path += '_' + dataset_normalisation

    # and the final error
    save_path += '_ac_' + str(np.round(score[1], 2))

    # and the final error
    save_path += '_l_' + str(np.round(score[0], 4))

    # mark the model complete, and then move it!
    save_path = save_path.replace('/incomplete/', '/complete/')
    os.rename(old_save_path, save_path)

    return models[-1], save_path


def save_history(dataset_name, save_path, history, prefix='', last_iteration=False, save_latex=False, save_csv=False):
    # save the raw history
    save_filename = prefix.replace(' ', '_') + '_' + dataset_name.replace(' ', '_') + '_model_history'

    pkl_save_file = save_filename + '.dill'
    if os.path.isfile(os.path.join(save_path, pkl_save_file)):
        with open(os.path.join(save_path, pkl_save_file), 'rb') as in_file:
            old_history = dill.load(in_file)
        for key in old_history:
            old_history[key].extend(history[key])
        history = old_history

    with open(os.path.join(save_path, pkl_save_file), 'wb') as out_file:
        dill.dump(history, out_file)

    if last_iteration:
        if save_csv:
            # also make a csv copy
            keys = sorted(history.keys())
            with open(os.path.join(save_path, save_filename + '.csv'), "wb") as out_file:
                writer = csv.writer(out_file, delimiter="\t")
                writer.writerow(keys)
                writer.writerows(zip(*[history[key] for key in keys]))

        if save_latex:
            # and finally save some of the information formatted for latex
            with open(os.path.join(save_path, save_filename + '.tex'), "wb") as out_file:
                line = '    '
                for key in history:
                    line += ' & ' + key
                line += ' \\\\ \\hline \n'
                out_file.write(line)

                line = 'best    '
                for key in history:
                    line += ' & '
                    if 'acc' in key:
                        line += str(max(history[key]))
                    else:
                        line += str(min(history[key]))
                line += ' \\\\ \\hline \n'
                out_file.write(line)

                line = 'last    '
                for key in history:
                    line += ' & ' + str(history[key][-1])
                line += ' \\\\ \\hline \n'
                out_file.write(line)

    return history


def copy_architecture_code(save_dir):
    network_file = 'networks.py'
    if os.path.exists(network_file):
        copyfile(network_file, os.path.join(save_dir, 'networks(copy).py'))
    else:
        raise Exception('Cannot find network file, has it been renamed? : %s ' % (network_file))


def save_network(save_dir, model, metadata):
    model.save(os.path.join(save_dir, 'model.h5'))
    with open(os.path.join(save_dir, 'model_metadata.dill'), 'wb') as save_file:
        dill.dump(metadata, save_file)
    copy_architecture_code(save_dir)


def load_network(directory):
    from tensorflow.keras.models import load_model
    model = load_model(os.path.join(directory, 'model.h5'))

    with open(os.path.join(directory, 'model_metadata.dill'), 'rb') as in_file:
        metadata = dill.load(in_file)

    for key, value in metadata.items():
        if key.lower() == 'export_acquisitons':
            # spelling correction for older models...
            setattr(model, 'export_acquisitions', value)
        else:
            setattr(model, key.lower(), value)
    return model


def reshape_data(data):
    n_channels = len(data[0])
    data = collapse_array(data)
    data = data.reshape(-1, len(data[0]), n_channels, 1)
    input_shape = (len(data[0]), n_channels, 1)
    return data, input_shape


def analyse_model(model, dataset_name, data, labels, output_labels, save_dir='./models/', prefix=''):
    if not all([x in model.output_labels for x in output_labels]):
        raise Warning('Output labels are not a subset of model.output labels.\n'
                      'There will be metabolites that won\'t be quantified.')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = np.array(data)
    labels = np.array(labels)
    data, input_shape = reshape_data(data)

    predicted_labels = normalise_labels(model.predict(data, verbose=1, batch_size=32), model.output_normalisation)
    save_analytics(labels, predicted_labels, output_labels, dataset_name, save_dir, model.output_normalisation,  prefix=prefix)
    for plot_func in [plot_labels_vs_labels, plot_metabolite_error_dist]:
        plot_func(labels, predicted_labels, output_labels, dataset_name, save_dir, prefix=prefix)
    return save_dir


def plot_label_distribution(labels, save_dir, metabolite_names=None):
    plt.figure()
    plt.suptitle('Concentration check. Range [0,1]. N=%i' % (labels.shape[0]))
    for ii in range(labels.shape[1]):
        plt.subplot(1, labels.shape[1], ii + 1)
        plt.hist(labels[:, ii], bins=50)
        if metabolite_names:
            plt.title('%s' % (metabolite_names[ii]))
        plt.xlim([0, 1])
    plt.savefig(os.path.join(save_dir, 'train_and_test_label_distribution.png'))
    plt.close()


def plot_metabolite_error_dist(true_labels, predicted_labels, output_labels, dataset_name, save_dir, prefix=''):
    metabolite_names = output_labels
    n_cols = 3
    n_rows = int(np.ceil(len(metabolite_names) / float(n_cols)))
    fig, axes = plt.subplots(int(n_rows), int(n_cols), sharex=True, sharey=True, figsize=(19.2, 19.2), dpi=200)
    fig.suptitle(
        'Per metabolite concentration error distribution (hist + KDE). Test dataset: ' + dataset_name + ' n:' + str(
            len(true_labels)))
    axes = axes.flatten()
    fig.text(0.5, 0.04, 'Error', ha='center')

    total_err_dist = []

    for ii in range(0, len(metabolite_names)):
        axes[ii].set_title(metabolite_names[ii])
        error = predicted_labels[:, ii] - true_labels[:, ii]
        total_err_dist.extend(error)
        try:
            sns.distplot(error, ax=axes[ii]).set(xlim=[-1, 1])
        except np.linalg.LinAlgError:
            print('Singular matrix found in dist plot - not saving.')
            return

    plt.savefig(os.path.join(save_dir, dataset_name + prefix + '_metabolite_error_distribution.png'))
    plt.close()


def plot_labels_vs_labels(true_labels, predicted_labels, output_labels, dataset_name, save_dir=None, prefix=''):
    import warnings
    metabolite_names = output_labels
    n_cols = 3
    n_rows = int(np.ceil(len(metabolite_names) / float(n_cols)))
    fig, axes = plt.subplots(int(n_rows), int(n_cols), sharex=True, sharey=True, figsize=(19.2, 19.2), dpi=100)
    axes = axes.flatten()
    fig.suptitle(
        'Predicted labels vs actual labels per metabolite(x,y). Test dataset name:' + dataset_name + ' n:' + str(
            len(true_labels)))
    fig.text(0.5, 0.04, 'Actual label', ha='center')
    fig.text(0.04, 0.5, 'Predicted label', va='center', rotation='vertical')

    for ii in range(0, len(metabolite_names)):
        sort_index = np.argsort(true_labels[:, ii])
        axes[ii].plot([0, 1], [0, 1], label='true line')
        sns.regplot(y=predicted_labels[sort_index, ii], x=true_labels[sort_index, ii], ax=axes[ii])

        # plot slope analysis
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # ignores division by zero warnings, which will happen in the analysis
            slope, intercept, r_value, p_value, std_err = linregress(true_labels[sort_index, ii],
                                                                 predicted_labels[sort_index, ii])
        axes[ii].set_title('%s $R^2$:%.2f Sl:%.2f Int:%.2f p:%.2f Err:%.2f' % (
            metabolite_names[ii], r_value, slope, intercept, p_value, std_err))

        if ii == 0:
            axes[ii].legend(loc=2)
            axes[ii].set_xlim([0, 1])
            axes[ii].set_ylim([0, 1])

    if save_dir:
        plt.savefig(os.path.join(save_dir, dataset_name + prefix + '_pred_labels_vs_true_labels_per_metabolite.png'))
    else:
        plt.show()
    plt.close()


def crop_labels(dataset_name, dataset_output_labels, dataset_true_labels, dataset_predicted_labels):
    target_metabolites = get_target_metabolites(dataset_name, dataset_output_labels)

    true_labels = np.zeros((len(dataset_true_labels[:, 0]), len(target_metabolites)))
    predicted_labels = np.zeros((len(dataset_true_labels[:, 0]), len(target_metabolites)))

    # now we take a union of metabolites that are in the sample, and metabolites that lcm is able to quantifiy
    # we build the error matrix and re-normalise it this way.
    for count, metabolite_name in enumerate(target_metabolites):
        metabolite_index = dataset_output_labels.index(metabolite_name)
        true_labels[:, count] = np.array(dataset_true_labels[:, metabolite_index])
        predicted_labels[:, count] = np.array(dataset_predicted_labels[:, metabolite_index])

    # re-normalise the values, each row is a spectra/scan(prediction), each col is a metabolite
    true_labels = true_labels / np.sum(true_labels, 1)[:, None]
    predicted_labels = predicted_labels / np.sum(predicted_labels, 1)[:, None]

    return target_metabolites, true_labels, predicted_labels


def get_target_metabolites(dataset_name, dataset_output_labels):
    target_metabolites = {'S4': ['n-acetylaspartate', 'gaba'],
                          'G3': ['n-acetylaspartate', 'gaba', 'glutamine', 'glutamate'],
                          'G4v2_1250': ['n-acetylaspartate', 'gaba', 'glutamine', 'glutamate'],
                          'G4v2_2000': ['n-acetylaspartate', 'gaba', 'glutamine', 'glutamate']}
    if dataset_name in target_metabolites.keys():
        return target_metabolites[dataset_name]
    else:
        return dataset_output_labels


def save_analytics(true_labels, predicted_labels, output_labels, dataset_name, save_dir, label_norm, prefix=''):
    save_file = os.path.join(save_dir, 'analytics.dill')

    if os.path.exists(save_file):
        with open(save_file, 'rb') as file_stream:
            data = dill.load(file_stream)
    else:
        data = {}

    save_key = prefix + dataset_name
    if save_key not in data.keys():
        data[save_key] = {'output_labels': output_labels,
                          'predicted_labels': predicted_labels,
                          'true_labels': true_labels,
                          'label_norm': label_norm}
    else:
        raise Exception('Data conflict : Key ' + str(save_key) + ' already in analytics data.')

    with open(save_file, 'wb') as file_stream:
        dill.dump(data, file_stream)


def plot_history(history, filepath='', show_plot=False, prefix=''):
    history_keys = history.keys()

    plt.figure(figsize=(19.2, 10.8), dpi=200)  # 3840 x 2160
    plt.subplot(3, 1, 1)
    for key in history_keys:
        if 'acc' in key:
            plt.semilogy(history[key], label=key)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 2)
    for key in history_keys:
        if 'error' in key:
            plt.semilogy(history[key], label=key)
    plt.title('model accuracy')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 3)
    for key in history_keys:
        if 'loss' in key:
            plt.semilogy(history[key], label=key)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(filepath, prefix + 'metrics.png'))

    if show_plot:
        plt.show()
    else:
        plt.close()


def load_benchmark_datasets():
    test_basi = []
    test_basi.append(Basis.load_dicom('./Datasets/Benchmark/E1/MEGA_Combi_WS_ON/'))
    test_basi.append(Basis.load_dicom('./Datasets/Benchmark/E3/MEGA_Combi_WS_ON/'))
    test_basi.append(Basis.load_dicom('./Datasets/Benchmark/E4a/MEGA_Combi_WS_ON/'))
    test_basi.append(Basis.load_dicom('./Datasets/Benchmark/E4a/MEGA_Combi_WS_ON/'))
    test_basi_names = ['E1', 'E3', 'E4_A', 'E4_B']

    return test_basi, test_basi_names


def train_network(basis, model_name, epochs, label_norm, batch_size, n_train_spectra, metabolites,
                  export_acquisitions, export_datatype, n_validation_spectra=None):

    if not n_validation_spectra:
        n_validation_spectra = np.ceil(n_train_spectra * 0.2)

    save_root = SAVE_ROOT

    n_train_lw_samples = int(np.round(float(n_train_spectra) / len(basis)))
    n_validation_lw_samples = int(np.round(float(n_validation_spectra) / len(basis)))

    train_datasets = []
    for basi in basis:
        train_dataset = Dataset()
        train_dataset._name = 'train'
        train_dataset.basis = basi
        train_dataset.export_datatype = export_datatype
        train_dataset.export_acquisitions = export_acquisitions
        train_dataset.conc_normalisation = label_norm
        train_dataset.export_nu = False
        train_dataset.export_dss_peak = False
        train_dataset.add_adc_noise = True
        train_dataset.add_nu_noise = False
        train_dataset.generate_dataset(metabolites, n_train_lw_samples)
        train_datasets.append(train_dataset)

    benchmark_basis, benchmark_basis_names = load_benchmark_datasets()
    benchmark_datasets = []
    for basi, name in zip(benchmark_basis, benchmark_basis_names):
        td = Dataset()
        td._name = 'benchmark_' + name
        td.basis = basi
        td.copy_settings(train_dataset, test_dataset=True)
        td.copy_basis()
        benchmark_datasets.append([td])

    validation_datasets = []
    for basi in basis:
        validation_dataset = Dataset()
        validation_dataset.basis = basi
        validation_dataset.copy_settings(train_dataset)
        validation_dataset._name = 'validation_multi_lw'
        validation_dataset.generate_dataset(metabolites, n_validation_lw_samples)
        validation_datasets.append(validation_dataset)

    multi_dataset_train_loop(model_name, epochs, batch_size, label_norm,
                             save_dir=save_root + 'models/',
                             train_datasets=train_datasets,
                             test_datasets=benchmark_datasets + [validation_datasets],
                             dataset_normalisation=train_dataset.conc_normalisation)


def multi_dataset_train_loop(model_name, epochs, batch_size, label_norm,
                             save_dir=None,
                             train_datasets=None, test_datasets=None,
                             dataset_normalisation=None):

    def check_output_labels(model_output_labels, proposed_output_labels):
        if model_output_labels is None:
            return proposed_output_labels
        else:

            if not len(model_output_labels) == len(proposed_output_labels):
                raise Exception('Model output labels and proposed output labels are of different lengths! This means '
                                'that there are not the same number of metabolites in different datasets...')

            if not model_output_labels == proposed_output_labels:
                raise Exception('Export labels are not aligned between datasets. '
                            'Either they are out of order, or have different lengths.')

        return model_output_labels

    data = []
    labels = []

    if len(train_datasets) == 1:
        dataset_name = train_datasets[0].name()
    else:
        dataset_name = 'mixed_'

    # save the metabolite names that correlate to the network output labels
    train_model_output_labels = None
    if train_datasets:
        for train_dataset in train_datasets:
            t_data, t_labels, mo_labels = train_dataset.export_to_keras()
            data.extend(t_data)
            labels.extend(t_labels)
            train_model_output_labels = check_output_labels(train_model_output_labels, mo_labels)


    data = np.array(data)
    labels = np.array(labels)

    model, model_save_dir = conv_network(data,
                                         labels,
                                         train_model_output_labels,
                                         dataset_name,
                                         model_name,
                                         epochs,
                                         batch_size,
                                         label_norm=label_norm,
                                         save_directory=save_dir,
                                         dataset_normalisation=dataset_normalisation,
                                         export_acquisitions=train_datasets[0].export_acquisitions,
                                         export_datatype=train_datasets[0].export_datatype)

    model_save_dir = analyse_model(model, dataset_name, data, labels, train_model_output_labels,
                                   save_dir=model_save_dir,
                                   prefix='train')

    if test_datasets is not None:
        for td_group in test_datasets:
            test_model_output_labels = None
            data = []
            labels = []

            for td in td_group:
                t_data, t_labels, export_labels = td.export_to_keras(model_labels=train_model_output_labels)
                dataset_name = td.name()
                data.extend(t_data)
                labels.extend(t_labels)
                test_model_output_labels = check_output_labels(test_model_output_labels, export_labels)

            model_save_dir = analyse_model(model, dataset_name, data, labels, test_model_output_labels,
                                           save_dir=model_save_dir)


if __name__ == '__main__':
    startup()
