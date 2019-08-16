#!/usr/bin/env python2.7
import os
import pdb
import fnmatch
import numpy as np
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt


def plot_weights(model, layer):
    weights = np.squeeze(model.layers[layer].get_weights()[0])
    weights = np.reshape(weights, (np.size(weights, 1), np.size(weights, 0)))
    plt.plot(weights)
    plt.show()


def nice_rename(word):
    # rename vars for publication
    if word in ['?']:
        return word

    names = {'G4': 'E4', 'G4v2_1250': 'E4$_A$', 'G4v2_2000': 'E4$_B$',
            'S4': 'E1', 'G3': 'E3', 'lcm': 'LCModel', 'pyg': 'PyGamma',
            'fida':'FID-A', 'train': 'Train', 'valid': 'Validation'}

    return names[word]


def convert_mol_names(metabolite_names, mode='shorten'):
    # preferred short name should be in the first index in the array, e.g. for myo-inositol 'MyI' will be chosen over 'mi'
    names = {'N-Acetylaspartate': ['NAA'],
             'Creatine': ['Cr', 'cre'],
             'GABA': ['GABA'],
             'Choline-truncated': ['Cho'],
             'Glutamate': ['Glu'],
             'Glutamine': ['Gln'],
             'Glutathione': ['gsh'],
             'Glycine': ['Gly'],
             'Lactate': ['Lac'],
             'Myo-Inositol': ['MyI', 'mi', 'ins'],
             'N-Acetylaspartylglutamic': ['NAAG'],
             'NAAG-truncated-siemens': ['NAAG-SIE'],
             'Phosphocreatine': ['PCr', 'pch'],
             'Taurine': ['Tau'],
             'Water': ['H20'],
             'DSS': ['DSS'],
             'Alanine': ['ala'],
             'Aspartate': ['asp'],
             'Scyllo-Inositol': ['scyllo']}

    metabolite_names = [m_name.lower() for m_name in metabolite_names]

    if mode == 'verify':
        all_names = names.keys()
        all_names.extend(collapse_array(names.values()))
        all_names = [x.lower() for x in all_names]
        for name in metabolite_names:
            if name.lower() not in all_names:
                raise Exception('\n\nMetabolite name "'+name+'" not found in convert_mol_names.\n'
                                'Please check spelling or add the short and long form to the dictionary.\n\n'
                                'Available metabolite names:\n' + str(names.keys()))
        return

    new_names = []
    for m_name in metabolite_names:
        for key in names.keys():
            if m_name == key.lower() or m_name in [x.lower() for x in names[key]]:
                if mode == 'shorten':
                    new_names.append(names[key][0])
                elif mode == 'lengthen':
                    new_names.append(key)
                else:
                    raise Exception('Unknown conversion mode: ' + mode)
                break

    if len(new_names) != len(metabolite_names):
        pdb.set_trace()
        raise Exception('Conversion failed, number of output names does not match the number of input names.')

    return new_names


def normalise_labels(labels, normalisation):
    if normalisation == 'max':
        with np.errstate(invalid='ignore'):
            if labels.ndim == 1:
                labels= labels / np.max(labels)
            elif labels.ndim == 2:
                labels = (labels.T / np.max(labels, 1)).T
    elif normalisation == 'sum':
        with np.errstate(invalid='ignore'):
            if labels.ndim == 1:
                labels = labels / np.sum(labels)
            elif labels.ndim == 2:
                labels = (labels.T / np.sum(labels, 1)).T
    else:
        raise Exception(
            'Unknown output normalisation for model, please write a custom routine for ' + normalisation)

    # replace the NaNs that come from [0,0,...]/0 with 0's
    labels[np.isnan(labels)] = 0

    return labels


def matlabify_string(string):
    # for exporting to matlab format, some characters are invalid
    to_replace = [' ', '-', '+', '  ', '/', '&', '&']
    for r in to_replace:
        string = string.replace(r, '_')
    return string


def num_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU' or  x.device_type == 'XLA_GPU'])


def normalise_signal(signal):
    # the reason we expect multiple channels is to ensure that the normalisation is the same across all the FFTs.
    # this is important in the case of MEGA-PRESS where we have two acquisitions, where we want both to be normalise
    # to the same value otherwise as NAA is edited out, the Cr peak jumps a good few points!
    return signal / np.max([np.abs(np.real(signal)), np.abs(np.imag(signal)), np.abs(signal)])


def get_files(directory, file_extension, recursive=False):
    # https://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    if not os.path.exists(directory):
        raise Exception('Directory does not exits: ' + directory)

    matches = []
    if recursive:
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, file_extension):
                matches.append(os.path.join(root, filename))
    else:
       for filename in fnmatch.filter(os.listdir(directory), file_extension):
            matches.append(os.path.join(directory, filename))
    matches.sort()
    return matches


def collapse_array(array):
    new_array = []
    for d in array:
        for a in d:
            new_array.append(a)

    return np.array(new_array)
