#!/usr/bin/env python2.7
from datetime import datetime
from functools import partial
from tqdm import tqdm
import multiprocessing
import sobol_seq

from Utilities.utils import *


class Dataset(object):
    """
        A dataset is a collection of spectra for training, tesing or predicting.
            The dataset object contains methods for how to handle and export the data from the spectra objects.
    """
    def __init__(self):
        self.basis = None
        self.generation_date = datetime.now()
        self.spectra = []
        self.linewidth = None

        # export settings
        self._export_labels = []
        self.export_datatype = 'magnitude'  # 'real', 'imaginary', 'real and imaginary'
        self.export_nu = False
        self.export_acquisitions = None
        self.add_adc_noise = False
        self.conc_gen_method = 'sobol'  # 'random uniform'
        self.conc_normalisation = 'sum'  # max, sum, none
        self.mean_center_export = True

        self.high_ppm = -4.5
        self.low_ppm = -1
        self.n_fft_pts = 2048

        self._name = None
        self._save_counter = 0
        self._save_directory = None

    def copy_settings(self, source_dataset, test_dataset=False):
        fields_to_copy = ['export_datatype', 'export_nu', 'conc_gen_method', 'conc_normalisation',
                          'high_ppm', 'low_ppm', 'n_fft_pts', 'export_acquisitions']
        if not test_dataset:
            fields_to_copy.extend(['add_adc_noise'])
            if self._name and (('test' in self._name) or ('benchmark' in self._name)):
                raise Exception('Are you sure this isn\'t a test/benchmark dataset?')

        for field in fields_to_copy:
            setattr(self, field, getattr(source_dataset, field))

    def name(self):
        if self._name is None:
            name = ''
        else:
            name = self._name + '_'

        name += self.basis.source() + '_' + self.pulse_sequence()
        name += '_exp_dt_' + str(self.export_datatype).replace('(', '').replace(')', '').replace(',', '_').replace(' ','').replace('\'','')

        if self.export_nu:
            name += '_exp_nu_' + str(self.export_nu)
        if self.add_adc_noise:
            name += '_adc_noise_' + str(self.add_adc_noise)

        name += '_lw_' + str(self.linewidth)

        return name

    def get_save_folder_name(self, directory=None):
        if self._save_directory is None:
            if directory is None:
                raise Exception('First call to get_save_folder_name must include the parent directory')
            dirname = 'dataset_' + datetime.now().strftime('%a-%d-%b-%Y_%H:%M:%s') + \
                      '_' + self.name().replace(' ', '_') + \
                      '_' + self.generation_date.strftime("%H:%M_%d-%m-%Y") + \
                      '_' + 'lw:' + str(self.linewidth) + '_'

            for name in self.spectra[0].metabolite_names:
                dirname += name + '_'

            self._save_directory = os.path.join(directory, dirname.rstrip('_'))

        if not os.path.exists(self._save_directory):
            os.makedirs(self._save_directory)

        return self._save_directory

    def check(self):
        if len(self.spectra) == 0:
            raise Exception('Dataset has no spectra associated with it')
        if self.basis is None:
            raise Exception('Dataset basis is none, please set it!')
        if self.linewidth is None:
            raise Exception('Please set dataset linewidth')
        self.basis.check()
        if self.high_ppm is None:
            raise Exception('Please set a high_ppm for the dataset')
        if self.low_ppm is None:
            raise Exception('Please set a low_ppm for the dataset')
        if self.high_ppm > self.low_ppm:
            raise Exception('High ppm needs to be less than low ppm.. E.G. high_ppm = -5, low_ppm = 0 as the'
                            ' frequency axis is backwards.')

        if isinstance(self.export_datatype, basestring):
            self.export_datatype = [self.export_datatype]

        for ed in self.export_datatype:
            if ed.lower() not in ['real', 'imaginary', 'phase', 'magnitude', 'absolute', 'r', 'i', 'p', 'm', 'a']:
                raise Exception('Unrecognised export datatype: ' + ed)

        if len(self.export_datatype) > 3:
            raise Exception('Length of export datatypes is too long (%d), it shouldn\'t be greater than 3.'
                            % (len(self.export_datatype)))

        for ii in range(0, len(self.spectra)):
            self.spectra[ii].check()

        self.basis.check()

    def add_spectra(self, spectra):
        self.spectra.extend(spectra)

    def pulse_sequence(self):
        return self.basis.pulse_sequence()

    def copy_basis(self, acquisitions=None):
        if acquisitions is not None:
            spec_list = []
            for spectra in self.basis.spectra:
                if spectra.acquisition in acquisitions:
                    spec_list.append(spectra)
            self.add_spectra(spec_list)
        else:
            self.add_spectra(self.basis.spectra)
        self.linewidth = self.basis.spectra[0].linewidth
        self.check()

    def update_cache(self):
        for spectra in self.spectra:
            spectra._need_fft_cache_update = True

    def get_metabolite_names(self):
        return self.spectra[0].metabolite_names

    def prime_rescale_fft(self):
        self.spectra[0].rescale_fft(high_ppm=self.high_ppm, low_ppm=self.low_ppm, npts=self.n_fft_pts)
        for ii in range(len(self.spectra)):
            self.spectra[ii].update_zero_pad(self.spectra[0]._zero_pad)

    def regenerate_dataset(self, basis, start_index=0, end_index=None):
        # update the basis, and regenerate the same dataset from a new basis
        spectra = self.group_spectra_by_id()
        if end_index is None:
            end_index = len(spectra)

        if end_index > len(spectra):
            raise Exception('end_index is greater than the number of spectra')

        if start_index > len(spectra):
            raise Exception('start_index is greater than the number of spectra')

        new_spec = []
        for spec in spectra[start_index: end_index]:
            spectra_export = basis.export_combination(spec[0].concentrations,
                                                      spec[0].metabolite_names,
                                                      acquisitions=self.export_acquisitions)
            for spec_ex in spectra_export:
                spec_ex._adc_noise = spec[0]._adc_noise

            new_spec.extend(spectra_export)

        self.basis = basis
        self.spectra = new_spec
        self.check()

    def generate_dataset(self, metabolite_names, n_samples, overwrite=False, checkpoint=None, save_dir=None):
        if n_samples <= 0:
            raise Exception('n_samples must be greater than 0! (%d)' % (n_samples))

        if save_dir is not None and checkpoint is None:
            raise Exception('Save directory has been set, but checkpoint has not! Please set it. ')

        if checkpoint is not None and save_dir is None:
            raise Exception('If you are setting a checkpoint for saving, you must specisify where you want them to '
                            'be saved (directory=None)')
        if len(self.spectra):
            if overwrite:
                print('Overwriting current dataset, as overwrite set to true.')
                self.spectra = []
            else:
                raise Exception('This dataset already has spectra associated with it, either set overwrite=True or '
                                'make a new dataset with the same basis for a different config.')

        n_metabolites = len(metabolite_names)
        for spectra in self.basis.spectra:
            if len(spectra.metabolite_names) > 1:
                raise Exception(
                    'Spectra in basis set cannot have more than one metabolite in if you want to generate a dataset.')

            if np.min(spectra.nu()) > self.high_ppm:
                raise Exception('Spectra does not reach the required max frequency axis (%.2f) for export: %.2f' % (np.min(spectra.nu()), self.high_ppm))
            elif np.max(spectra.nu()) < self.low_ppm:
                raise Exception('Spectra does not reach the required max frequency axis (%.2f) for export: %.2f' % (np.max(spectra.nu()), self.low_ppm))

        if n_metabolites > len(self.basis.spectra):
            raise Exception('Number of metabolite names is greater than the number of spectra in the basis.')
        if len(metabolite_names) != len(set(metabolite_names)):
            raise Exception('At least one metabolite name appears more than once in the list: ' + str(metabolite_names))

        # filter the spectra to make sure it only contains metabolites in the metabolite names list, or a
        # combination of them any additional metabolites are removed
        spectra_to_keep = []
        for ii in range(0, len(self.basis.spectra)):
            if len(self.basis.spectra[ii].metabolite_names) <= len(metabolite_names):
                # checks if every metabolite name in the spectra is also in the metabolite_names list
                if all([smn in metabolite_names for smn in self.basis.spectra[ii].metabolite_names]):
                    spectra_to_keep.append(self.basis.spectra[ii])
        if not len(spectra_to_keep):
            raise Exception('No spectra selected to keep!')

        self.basis.spectra = spectra_to_keep
        self.basis.setup()

        # now we've prepped the basis set time to set up the dataset.
        self.linewidth = self.basis.spectra[0].linewidth

        if self.conc_gen_method == 'random all':
            # verstion where all metabolites can be excited
            concentrations = np.random.ranf((n_samples, n_metabolites))
            concentrations = (concentrations.T / np.sum(concentrations, 1)).T
        elif self.conc_gen_method == 'random partial':
            # version where not all metabolites are excited
            concentrations = np.zeros((n_samples, n_metabolites))
            # decide how many metabolites will be excited for each concentraion
            n_excited_samples = np.random.randint(1, n_metabolites + 1, size=n_samples)
            # concentrations for each of the aformentioned metabolites
            excited_samples = [sorted(np.random.choice(range(n_metabolites), n_s, replace=False)) for n_s in
                               n_excited_samples]

            for c, e in zip(concentrations, excited_samples):
                c[e] = np.random.ranf(len(e))
                # normalise the concentration to be a percentage for each (softmax output from CNN) (sum = 1)
            concentrations = (concentrations.T / np.sum(concentrations, 1)).T
        elif self.conc_gen_method == 'random uniform':
            # version where there is uniform sampling across the number of excited metabolites
            concentrations = np.zeros((n_samples, n_metabolites))
            n_samples_per_div = int(np.ceil(n_samples / float(n_metabolites)))
            samples = []
            for n_excited_metabolites in range(1, n_metabolites + 1):
                samples.extend(
                    [sorted(np.random.choice(range(n_metabolites), n_excited_metabolites, replace=False)) for _ in
                     range(n_samples_per_div)])
            for ii, e in zip(range(n_samples), samples):
                concentrations[ii][e] = np.random.ranf(len(e))
        elif self.conc_gen_method == 'sobol':
            concentrations = sobol_seq.i4_sobol_generate(n_metabolites, n_samples)
        else:
            raise Exception('Unknown concentration generation method: ' + self.conc_gen_method)

        for count in range(n_samples):
            # then export the combination and add it to the dataset
            self.add_spectra(self.basis.export_combination(concentrations[count],
                                                           metabolite_names,
                                                           acquisitions=self.export_acquisitions))
            # checkpointing for saving large dataset objects, if specified
            if checkpoint is not None and count > 0:
                if (count % checkpoint == 0) or (count == (n_samples - 1)):
                    self.check()
                    spectra, labels, export_labels = self.export_to_keras()
                    self.save_compressed(save_dir, spectra, labels)
                    self.spectra = []

        if checkpoint is None:
            self.check()
        else:
            return self.get_save_folder_name()

    def group_spectra_by_id(self):
        ids = []
        for ii in range(0, len(self.spectra)):
            ids.append(self.spectra[ii].id)

        n_acquisitions = ids.count(ids[0])
        ids = list(set(ids))

        acquisitions = [np.array([]) for _ in range(len(ids))]

        # group the acqusitions by ID, and also sort them by acquisition
        for count, id in enumerate(ids):
            acquisition_numbers = []
            for ii in range(0, len(self.spectra)):
                if self.spectra[ii].id == id:
                    if self.export_acquisitions is not None:
                        # if the acqusitions are in the list of ones to be exported
                        if self.spectra[ii].acquisition in self.export_acquisitions:
                            acquisition_numbers.append(self.spectra[ii].acquisition)
                            acquisitions[count] = np.append(acquisitions[count], self.spectra[ii])
                    else:
                        acquisition_numbers.append(self.spectra[ii].acquisition)
                        acquisitions[count] = np.append(acquisitions[count], self.spectra[ii])

                    if len(acquisition_numbers) == n_acquisitions:
                        # we've found all the acquisitions for this ID
                        break

            # sort the acquisitions
            acquisitions[count] = acquisitions[count][np.argsort(acquisition_numbers)]

        return acquisitions

    def export_to_keras(self, model_labels=None):
        if model_labels is not None:
            if isinstance(model_labels, np.ndarray):
                model_labels = model_labels.tolist()
            elif not isinstance(model_labels, list):
                raise Exception('model_labels needs to be of type list or np.ndarray.')

        # firstly copy the export settings to the individual spectra and check
        for ii in range(0, len(self.spectra)):
            self.spectra[ii].add_adc_noise = self.add_adc_noise
        self.check()

        # now double check the adc noise sigma & mu values across the acquisitions
        if self.add_adc_noise:
            for group in self.group_spectra_by_id():
                for spectra in group:
                    spectra._adc_noise_sigma = group[0]._adc_noise_sigma
                    spectra._adc_noise_mu = group[0]._adc_noise_mu
                    spectra.generate_adc_noise(overwrite=True)

        # organise the output mapping for the metabolite names and the concentrations
        if model_labels is None:
            model_labels = []
            for spectra in self.spectra:
                model_labels.extend(spectra.metabolite_names)
            model_labels = sorted(list(set(model_labels)))

        for ii in tqdm(range(len(self.spectra)), desc='Verifying concentration mappings', leave=False, total=len(self.spectra) - 1):
            self.spectra[ii].metabolite_names = np.array(self.spectra[ii].metabolite_names)
            self.spectra[ii].concentrations = np.array(self.spectra[ii].concentrations)

            if (not len(self.spectra[ii].metabolite_names) == len(model_labels)) or \
                    (not all(self.spectra[ii].metabolite_names == model_labels)):
                # either the ordering is wrong, or there are metabolites missing.
                # then lets replace the concentration array length to accommodate this, while mapping the old
                # concentrations to the new ones.

                # first we get the indexes of metabolites that are present, in relation to model_labels
                index = []
                for name in [x.lower() for x in self.spectra[ii].metabolite_names]:
                    if name in [x.lower() for x in model_labels]:
                        index.append([x.lower() for x in model_labels].index(name))

                # now we create a new label array, with the correct model->label arrangement
                new_conc = np.zeros(len(model_labels))
                for c, i in enumerate(index):
                    new_conc[i] = self.spectra[ii].concentrations[c]

                # finally update the spectra labels & model lables (no need to regen the FFTs as it's the same)
                self.spectra[ii].concentrations = new_conc
                self.spectra[ii].metabolite_names = model_labels

        self._export_labels = model_labels

        # in theory, we should only have to do the zero_pad calculation once
        # it checks the length of the zero pad needed on one of the spectra and copies it to all of the output
        self.prime_rescale_fft()

        # now we group all of the spectra together by ID and in the correct order of acqusition
        acquisitions = self.group_spectra_by_id()

        if len(acquisitions) < 200:
            data = []
            labels = []
            for acq in acquisitions:
                temp_data, temp_labels = parallel_export_function(acq, high_ppm=self.high_ppm, low_ppm=self.low_ppm,
                                                        n_fft_pts=self.n_fft_pts, export_datatype=self.export_datatype,
                                                        export_nu=self.export_nu, mean_center=self.mean_center_export)
                data.append(temp_data)
                labels.append(temp_labels)
        else:
            try:
                pool = multiprocessing.Pool()

                func = partial(parallel_export_function,
                               high_ppm=self.high_ppm,
                               low_ppm=self.low_ppm,
                               n_fft_pts=self.n_fft_pts,
                               export_datatype=self.export_datatype,
                               export_nu=self.export_nu,
                               mean_center=self.mean_center_export)

                data, labels = zip(*list(tqdm(pool.imap(func, iterable=acquisitions, chunksize=32),
                                              desc='Exporting data',
                                              total=len(acquisitions),
                                              leave=False)))
                pool.close()
                pool.join()

            except:
                print('\n\nFound error in parallel export function, running without multi-processing pool to find and '
                      'raise the underlying error:\n')
                for ii in range(len(acquisitions)):
                    data, labels = parallel_export_function(acquisitions[ii], high_ppm=self.high_ppm, low_ppm=self.low_ppm,
                                                            n_fft_pts=self.n_fft_pts, export_datatype=self.export_datatype,
                                                            export_nu=self.export_nu, mean_center=self.mean_center_export)

        if len(np.shape(data)) != 3:
            raise Exception('Export does not have dim(3), when it should...')

        labels = normalise_labels(np.array(labels), self.conc_normalisation)

        data = np.array(data)

        return data, labels, self._export_labels


def parallel_export_function(acquisition, high_ppm, low_ppm, n_fft_pts,
                             export_datatype, export_nu=False, mean_center=True):
    # map the spectra metabolite names to the correct output labels on the given network.
    labels = acquisition[0].concentrations
    for acq in acquisition:
        if not all(acq.concentrations == labels):
            raise Exception('Concentrations are not consistent across acqusitions.')

    ffts = []
    # gather all the ffts for the acquisitions
    for jj in range(len(acquisition)):  # for each acquisiton of each sample
        fft, nu = acquisition[jj].rescale_fft(high_ppm=high_ppm, low_ppm=low_ppm, npts=n_fft_pts)

        if mean_center:
            fft = fft - np.mean(fft)

        ffts.append(fft)

    # we normalise the signals once we have them all as we want them relative to each other
    # this could be more compact, but keeping it expanded out for clarity
    # the idea here is to try and preserver the order of the acqusitions e.g:
    #           | real      |
    #  acq 0 :  | imaginary |
    # __________| magnitude |
    #           | real      |
    #  acq 1 :  | imaginary |
    # __________| magnitude |
    #           | real      |
    #  acq 2 :  | imaginary |
    #           | magnitude |

    spec_data = []
    for fft in ffts:
        if any([x in export_datatype for x in ['r', 'real']]):
            spec_data.append(np.real(fft))
        if any([x in export_datatype for x in ['i', 'imaginary']]):
            spec_data.append(np.imag(fft))
        if any([x in export_datatype for x in ['m', 'magnitude', 'a', 'absolute']]):
            spec_data.append(np.abs(fft))

    # normalse the spectra data to fall in the -1:1 range
    spec_data = spec_data / np.max(np.abs(spec_data))

    if any(np.isnan(np.ndarray.flatten(np.array(spec_data)))):
        raise Exception('"spec_data" export array contains NaNs. Something has gone wrong here...')

    if export_nu:
        spec_data.append(nu)

    return spec_data, labels
