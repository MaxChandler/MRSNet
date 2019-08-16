#!/usr/bin/env python2.7
import dill
import random
import Utilities.util_dicom_siemens as util_dicom_siemens

from scipy.io import loadmat
from scipy import signal
from Utilities.utils import *
from Utilities.constants import NAA_REFERENCE, CR_REFERENCE, WATER_REFERENCE, DICOM_METADATA, GYROMAGNETIC_RATIO


class Spectra(object):
    """
        Spectra is a class that contains information about one single spectrum loaded from any source.
            Time domain data is preferred over frequency domain when loading.
    """
    def __init__(self):
        self.id = None
        self.source = None              # pygamma, fida, lcmodel,...
        self.type = None                # simulated or real
        self.pulse_sequence = None      # megapress, press, steam, fid, etc...
        self.spectra_type = None        # for megapress: edit on, edit off, difference
        self.metabolite_names = []
        self.concentrations = []
        self.scale = 1
        self.center_ppm = 0
        self.omega = None
        self.dt = None
        self.sw = None
        self.te = None
        self.linewidth = None
        self.metadata = {}
        self.acquisition = None         # for basis sets with multiple acquisitons of the same metabolite, e.g.
                                        # MEGA-PRESS with edit on/off

        self.add_adc_noise = False
        self.remove_water_peak = True

        self._source_filepath = None
        self._filter_fft = False        # this is only applied by default to dicom files
        self._raw_adc = []
        self._zero_pad = 0
        self._adc_noise = None

        self._correct_b0 = True         # do we allow b0 correction for this spectra?
        self._b0_reference_frequency = None
        self._b0_ppm_shift = 0

        self._b0_fft_pts_shift = 0      # explanation for these two can be found in correct_b0
        self._b0_nu_shift = 0

        self._adc_noise_mu = None
        self._adc_noise_sigma = None

        self._fft_cache = None
        self._need_fft_cache_update = False

    def check(self):
        self.convert_mol_name()

        if not len(self._raw_adc):
            raise Exception('Spectra had no raw adc associalted with it')

        if self.acquisition is None:
            raise Exception('Please set acqusition number for spectra.')

        if self.type is None:
            raise Exception('Please set type of spectra')

        if self.spectra_type is None and self.pulse_sequence == 'megapress':
            raise Exception('Spectra pulse sequence is megapress but spectra type is not set, please set it!')

        if self.pulse_sequence == 'megapress' and self.spectra_type not in ['edit off', 'edit on', 'difference']:
            raise Exception(
                'Pulse sequence is megapress, but spectra_type not in [edit off, edit on, difference]: ' + self.spectra_type)

        if self._adc_noise is None and self.add_adc_noise:
            self.generate_adc_noise()

        if any(np.isnan(self._raw_adc)):
            raise Exception('raw adc contains nan values')

    def zero_signal(self):
        return np.zeros(len(self._raw_adc) + self._zero_pad, dtype=complex)

    def adc(self):
        if self.scale > 0:
            adc = self.scale * self._raw_adc

            if self.add_adc_noise:
                adc += self._adc_noise  # (self._adc_noise * np.max(self._raw_adc))

            adc = np.append(adc, np.zeros(self._zero_pad))

        elif self.scale == 0:
            adc = self.zero_signal()
        else:
            raise Exception('Spectra scale cannot be less than 0')
        return adc

    def nu(self, npts=None):
        if npts is None:
            npts = len(self.adc())

        if self.source == 'dicom':
            ppm_range = (self.sw / self.omega) / 2
            nu = np.linspace(-ppm_range, ppm_range, npts) + self.center_ppm
        elif self.source in ['pygamma', 'fida', 'lcmodel']:
            nu = ((np.linspace(-1, 1, npts) * (1 / self.dt / 2)) / self.omega) + self.center_ppm
        else:
            raise Exception('Please write custom nu routine for input source: ' + self.source)

        # finally apply the fine grained b0 correction, this is on the order of < delta_nu
        nu += self._b0_nu_shift

        return nu

    def delta_nu(self):
        return np.abs(self.nu()[0] - self.nu()[1])

    def ppm_to_nu_pts(self, ppm_shift):
        return int(np.floor(ppm_shift / self.delta_nu()))

    def nu_pts_to_ppm(self, n_nu_pts):
        return self.delta_nu() * n_nu_pts

    def fft(self, update_cache=False):
        if not update_cache:
            # do some checking here, just in case we do need to forcibly recalculate the fft
            if self._need_fft_cache_update:
                update_cache = True
            elif self._fft_cache is None:
                update_cache = True
            elif len(self.adc()) != len(self._fft_cache):
                update_cache = True

        if update_cache:
            self._need_fft_cache_update = False
            if self.scale != 0:
                adc = self.adc()
                # fft routines for different input sources
                if self.source == 'dicom' or self.source == 'lcmodel':
                    fft = np.fft.fftshift(np.fft.fft(adc, len(adc)))
                elif self.source == 'pygamma' or self.source == 'fida':
                    fft = np.flip(np.fft.fftshift(np.fft.fft(adc, len(adc)), 0))
                else:
                    raise Exception('Please write custom raw_fft routine for input source: ' + self.source)

                if self._filter_fft:
                    b, a = signal.butter(1, 0.7)
                    fft = signal.filtfilt(b, a, fft, padlen=150)
            else:
                fft = self.zero_signal()

            # b0 correction is required
            if self._b0_fft_pts_shift != 0:
                if self._b0_fft_pts_shift > 0:
                    fft = np.pad(fft, (0, self._b0_fft_pts_shift), 'constant', constant_values=np.mean(fft))[
                          self._b0_fft_pts_shift:]
                elif self._b0_fft_pts_shift < 0:
                    fft = np.pad(fft, (np.abs(self._b0_fft_pts_shift), 0), 'constant', constant_values=np.mean(fft))[
                          :self._b0_fft_pts_shift]

            if self.type == 'real' and self.remove_water_peak:
                fft = Spectra.remove_water_peak(fft, self.nu(), ppm_range=1)

            self._fft_cache = fft

        return self._fft_cache

    def raw_fft(self):
        # no scaling, zero padding, shifting or correction at all
        if self.source == 'dicom' or self.source == 'lcmodel':
            return np.fft.fftshift(np.fft.fft(self._raw_adc, len(self._raw_adc)))
        elif self.source == 'pygamma' or self.source == 'fida':
            return np.flip(np.fft.fftshift(np.fft.fft(self._raw_adc, len(self._raw_adc)), 0))
        else:
            raise Exception('Please write custom raw_fft routine for input source: ' + self.source)

    def convert_mol_name(self):
        self.metabolite_names = convert_mol_names(self.metabolite_names, mode='lengthen')

    def plot(self):
        fft, nu = self.rescale_fft()
        figure = plt.figure()
        n_cols = 1
        super_title = ''
        if len(self.concentrations) > 1:
            n_cols = 2
        else:
            super_title += self.metabolite_names[0] + ' '

        super_title += ' Source: ' + self.source
        super_title += ' Pulse sequence: ' + self.pulse_sequence

        if self.spectra_type is not None:
            super_title += ' Spec. type: ' + str(self.spectra_type)

        plt.suptitle(super_title)

        plt.subplot(3, n_cols, 1)
        plt.title('Absolute')
        plt.plot(nu, np.abs(fft))
        plt.xlim([-4.5, -1])

        plt.subplot(3, n_cols, 1 + n_cols)
        plt.title('Real')
        plt.plot(nu, np.real(fft))
        plt.xlim([-4.5, -1])

        plt.subplot(3, n_cols, 1 + (n_cols * 2))
        plt.title('Imaginary')
        plt.plot(nu, np.imag(fft))
        plt.xlim([-4.5, -1])
        # to do, reverse sign on axis

        if n_cols == 2:
            plt.subplot(1, 2, 2)
            plt.title('Concentrations:' + str(self.metabolite_names))
            plt.bar(np.linspace(0, len(self.concentrations) - 1, len(self.concentrations)), self.concentrations)
        plt.show()
        return figure

    def get_max_metabolite_amplitude(self, magnitude=False):
        metabolite_range = [-4.2, -1.5]
        fft = self.fft()
        nu = self.nu()

        m_fft = fft[(nu < np.max(metabolite_range)) & (nu > np.min(metabolite_range))]

        if magnitude:
            return np.max(np.abs(m_fft))
        else:
            return np.max([np.abs(np.real(m_fft)), np.abs(np.imag(m_fft))])

    def generate_adc_noise(self, overwrite=False):
        # these were intially stored with the spectra, but that meant they weren't as dynamic as I'd like.

        mu_choices = [0]
        sigma_choices = np.linspace(0.0, 0.25, 50)
        noise_chance = 0.5

        if self._adc_noise is None or overwrite:
            if self._adc_noise_mu is not None and self._adc_noise_sigma is not None:
                # this is for when mu and sigma are supplied, then we 100% want noise, else leave it to chance.
                self._adc_noise = np.random.normal(self._adc_noise_mu, self._adc_noise_sigma, len(self._raw_adc)) + \
                                  (1j * np.random.normal(self._adc_noise_mu, self._adc_noise_sigma, len(self._raw_adc)))
            else:
                if random.random() > noise_chance:
                    # so we make a choice for mu and sigma and store that option.
                    if self._adc_noise_mu is None:
                        self._adc_noise_mu = random.choice(mu_choices)
                    if self._adc_noise_sigma is None:
                        self._adc_noise_sigma = random.choice(sigma_choices)

                    # then generate the noise from those choices
                    self._adc_noise = np.random.normal(self._adc_noise_mu, self._adc_noise_sigma, len(self._raw_adc)) + \
                                      (1j * np.random.normal(self._adc_noise_mu, self._adc_noise_sigma, len(self._raw_adc)))
                else:
                    self._adc_noise = np.zeros(len(self._raw_adc))
        else:
            raise Exception('Tried to overwrite adc noise, this is not allowed by default. '
                            'If you want to do this, pass overwrite=true')

    def get_adc_noise(self):
        if self._adc_noise is None:
            self.generate_adc_noise()
        return self._adc_noise

    @staticmethod
    def load(filepath):
        with open(filepath) as in_file:
            return dill.load(in_file)

    def save(self, directory):
        self.check()
        filename = '-'.join(self.metabolite_names) + 'acq_' + self.acquisition + '.dill'
        with open(os.path.join(directory, filename), 'wb') as out_file:
            dill.dump(self, out_file)

    def load_metadata(self):
        # matches the ima files to the dict of metadata in Utilities->constants.property
        # matching is based on the filepath
        split_filepath = self._source_filepath.split('/')
        series_name = [x for x in split_filepath if x in DICOM_METADATA.keys()]
        found = False
        # first we find which series
        if len(series_name)  == 1:
            series_name = series_name[0]
            spectra_id = [x for x in split_filepath if x in DICOM_METADATA[series_name].keys()]
            if len(spectra_id) == 1:
                # found it! Now add the data to self
                spectra_id = spectra_id[0];
                found = True
                if 'b0_ppm_shift' in DICOM_METADATA[series_name][spectra_id].keys():
                    self.correct_b0(DICOM_METADATA[series_name][spectra_id]['b0_ppm_shift'])
                if 'concentrations' in DICOM_METADATA[series_name][spectra_id].keys():
                    self.concentrations = [float(i) for i in DICOM_METADATA[series_name][spectra_id]['concentrations']]
                if 'metabolite_names' in DICOM_METADATA[series_name][spectra_id].keys():
                    self.metabolite_names = DICOM_METADATA[series_name][spectra_id]['metabolite_names']
                    if self.metabolite_names:
                        self.id = '_'.join(self.metabolite_names) + '_' + self.metadata['patient_id'] + '_' + spectra_id
                if self.pulse_sequence == 'megapress':
                    self.id = spectra_id
            elif len(spectra_id) > 1:
                raise Exception('Spectra matched more than one ID: ' + spectra_id)
            else:
                raise Exception('Spectra matched series, but did not match any ID: ' + self._source_filepath)
        elif len(series_name) > 1:
            raise Exception('Dicom data matches more than one series (match done on foler names).\nMatching series: ' + series_name +'\nIMA Filepath: '+self._source_filepath)
        elif len(series_name) == 0:
            # no data for the dicom was found
            pass

        if self.pulse_sequence == 'megapress':
            if 'EDIT_OFF' in self._source_filepath:
                self.spectra_type = 'edit off'
                self.acquisition = 0
            elif 'EDIT_ON' in self._source_filepath:
                self.spectra_type = 'edit on'
                self.acquisition = 1
            elif 'DIFF' in self._source_filepath:
                self.spectra_type = 'difference'
                self.acquisition = 2
            else:
                # There is no good reliable way to identify these without guessing based off the time and dates, then working backwards.
                raise Exception('Loaded dicom file of type MEGA-PRESS, but I can\'t figure out which acquisition this '
                                'is (Edit On, Edit Off or Difference). \n'
                                'Please manuall specifiy it (add "EDIT_OFF", "EDIT_ON" or "DIFF" into the filepath anywhere).')

        if self.pulse_sequence == 'megapress' and not found:
            raise Exception('Spectra ID has not been set successfully for spectra. \n This is required for MEGA-PRESS '
                            'spectra to group them properly. Please add a unique ID to your dicom file names, and add '
                            'this ID to the dictionary in Utilties.constants to group the scans together. \n'
                            'See Utilities.constants and Spectra.load_dicom for more information.')

    @staticmethod
    def load_dicom(ima_file):
        if not os.path.exists(ima_file):
            raise Exception('ima file does not exist: ' + ima_file)

        spectra = Spectra()
        dicom_data = util_dicom_siemens.read(ima_file)
        spectra._source_filepath = ima_file
        spectra.metadata = dicom_data[0]
        spectra.source = 'dicom'
        spectra.id = ima_file.split('/')[-1]
        spectra._raw_adc = [np.array(dicom_data[1])][0]
        spectra.sw = float(spectra.metadata['sweep_width'])
        spectra.omega = float(spectra.metadata['frequency']) / 1e+6
        spectra.dt = -(spectra.omega / spectra.sw) / 1e+2
        spectra.center_ppm = -4.7
        spectra.pulse_sequence = spectra.metadata['sequence_filename'].rstrip('.').lstrip('*')
        spectra.te = float(spectra.metadata['echo_time'])
        spectra.acquisition = 0
        spectra.type = 'real'
        spectra.linewidth = -1      # Unknown!
        spectra._filter_fft = True

        # do some translation of the pulse sequence types - this works for Siemens
        if spectra.pulse_sequence in ['svs_ed', 'megapress']:
            spectra.pulse_sequence = 'megapress'
        elif spectra.pulse_sequence == 'svs_se':
            spectra.pulse_sequence = 'press'
        elif spectra.pulse_sequence == 'svs_st':
            spectra.pulse_sequence = 'steam'
        else:
            raise Exception(
                'Unrecognised dicom pulse sequence: ' + spectra.pulse_sequence + '. Please add an \'elif\' statement for it. ')

        spectra.load_metadata()
        spectra.check()

        return spectra


    @staticmethod
    def load_fida(fida_file):
        spectra = Spectra()
        spectra.source = 'fida'
        spectra.type = 'simulated'
        fida_data = loadmat(fida_file)
        spectra._source_filepath = fida_file
        spectra.pulse_sequence = 'megapress'
        spectra.metabolite_names = [str(fida_data['m_name'][0])]
        spectra.convert_mol_name()
        spectra.dt = np.abs(fida_data['t'][0][0] - fida_data['t'][0][1])
        spectra._raw_adc = np.transpose(fida_data['fid'])[0]
        spectra.omega = float(fida_data['omega'][0][0]) * GYROMAGNETIC_RATIO
        spectra.linewidth = float(fida_data['linewidth'][0][0])
        spectra.center_ppm = -np.median(fida_data['nu'])
        spectra.id = spectra.metabolite_names[0]

        if bool(fida_data['edit'][0][0]):
            spectra.acquisition = 1
            spectra.spectra_type = 'edit on'
        else:
            spectra.acquisition = 0
            spectra.spectra_type = 'edit off'

        spectra.check()
        return spectra

    @staticmethod
    def load_lcm_basis(file_buffer, dt, omega):
        # http://s-provencher.com/pub/LCModel/manual/manual.pdf
        spectra = Spectra()
        spectra.source = 'lcmodel'
        spectra.type = 'simulated'
        spectra.dt = dt
        spectra.omega = omega
        spectra.linewidth = 1
        area = None
        var_buffer = ''

        for counter, line in enumerate(file_buffer):
            if '$NMUSED' in line:
                area = 'nmused'
                continue
            elif "$BASIS" in line:
                area = 'basis'
                continue
            elif '$END' in line:
                # automatically assume that the next area is the spectra region, it may be the basis section
                if len(var_buffer):
                    spectra.add_lcm_metadata(var_buffer)
                    var_buffer = ''
                area = 'spectra'
                continue

            # var buffer is used as some variables span multiple lines, the only break is either commas or $END
            # there's also no end marker to the spectra in most cases
            var_buffer += line
            if ',' in var_buffer or (len(file_buffer) - 1) == counter:
                if area == 'nmused' or area == 'basis':
                    spectra.add_lcm_metadata(var_buffer)
                elif area == 'spectra':
                    spectra.add_lcm_fft(var_buffer)
                else:
                    raise Exception('No area set ' + str(counter) + ' :' + var_buffer)
                var_buffer = ''

        if 'PPMSEP' in spectra.metadata:
            spectra.center_ppm = -spectra.metadata['PPMSEP']
        else:
            spectra.center_ppm = -4.65      # defualt LCM value

        # Convert the LCM basis set names to something more universal, see utils.convert_mol_names.
        spectra.metabolite_names = convert_mol_names([spectra.metadata['METABO'].replace('\'', '')], mode='lengthen')
        spectra.acquisition = 0
        spectra.check()
        return spectra

    def add_lcm_metadata(self, var_buffer):
        if '=' in var_buffer:
            for to_remove in [' ', ',', '   ', '\n']:
                var_buffer = var_buffer.replace(to_remove, '')
            split_line = var_buffer.split('=')
            if split_line[0] in ['CONCSC', 'PPMPK', 'PPMPHA', 'PPMSCA', 'PPMSEP', 'PPMSCA', 'HWDPHA', 'HWDSCA',
                                 'PPMBAS', 'FWHMSM', 'CONCSC', 'CONC']:
                self.metadata[split_line[0]] = float(split_line[1])
            elif split_line[0] in ['NDATAB', 'ISHIFT']:
                self.metadata[split_line[0]] = int(split_line[1])
            elif split_line[0] in ['AUTOPH', 'AUTOSC', 'SCALE1', 'CONSISTENT_SCALING', 'NOSHIF', 'DO_CONTAM', 'FLATEN']:
                if 'F' in split_line[1]:
                    self.metadata[split_line[0]] = False
                elif 'T' in split_line[1]:
                    self.metadata[split_line[0]] = False
                else:
                    raise IOError('Unknown boolean(?) type: ' + split_line[1])
            else:
                # not sure what to do with these lines, keep them as strings...?
                self.metadata[split_line[0]] = split_line[1]
        else:
            raise Exception('Not sure how to handle line :' + var_buffer)

    def add_lcm_fft(self, var_buffer):
        fft = []
        nums = var_buffer.replace('\n', '').split()

        if len(nums) % 2 != 0:
            raise Exception('Uneven fft number, the real/imag switching does not work here or the file has been loaded '
                            'in wrong!')

        for ii in range(0, len(nums), 2):
            fft.append(float(nums[ii]) + (1j * float(nums[ii + 1])))

        # all the lcmodel spectra are stored as fourier transforms, so we convert them back to the ADC
        self._raw_adc = np.flip(np.fft.fft(fft), 0)

    @staticmethod
    def get_peak_location(fft, nu, location, ppm_range):
        # finds the highest peak from location +- ppm_range
        peaks = (-np.abs(fft)).argsort()[:len(nu) / 100]
        for peak in nu[peaks]:
            if location + ppm_range >= peak >= location - ppm_range:
                return peak
        return None

    @staticmethod
    def get_water_peak_location(fft, nu):
        return Spectra.get_peak_location(fft, nu, WATER_REFERENCE, 0.5)

    @staticmethod
    def get_naa_peak_location(fft, nu):
        return Spectra.get_peak_location(fft, nu, NAA_REFERENCE, 0.5)

    @staticmethod
    def remove_water_peak(fft, nu, ppm_range=0.6):
        # find the peak then set the range centered around it to the median signal of the fft
        water_peak_loc = Spectra.get_water_peak_location(fft, nu)
        ppm_range = float(ppm_range)
        mean_abs = np.mean(np.abs(fft))
        abs_fft = np.abs(fft)
        under_mean = 0
        if water_peak_loc is not None:
            mean = np.mean(np.real(fft)) + 1j * np.mean(np.imag(fft))
            for jj in range(0, len(fft)):
                if (water_peak_loc - (ppm_range / 2) < nu[jj]) and (water_peak_loc + (ppm_range / 2) > nu[jj]):
                    if abs_fft[jj] > mean_abs:
                        under_mean = 0
                        fft[jj] = mean
                    if nu[jj] > water_peak_loc and abs_fft[jj] < mean_abs:
                        # trailing edge detection, clip it as soon as the water peak (in absolute terms is over).
                        under_mean += 1
                        if under_mean > 5:
                            return fft
        return fft

    def update_zero_pad(self, new_zero_pad):
        if new_zero_pad != self._zero_pad:
            self._zero_pad = new_zero_pad
            if self._b0_ppm_shift:
                self.update_b0_shift(self._b0_ppm_shift)

    def correct_b0(self, ppm_shift=None):
        # the way this works is twofold
        # the major B0 correction will be done by padding and trimming the fft, shifting it on the frequency axis nu
        # the issue is that this then only has a granularity of delta_nu, so the remaining b0 shit has to be corrected
        # by offsetting the entire nu axis by a value < delta_nu
        if self._correct_b0:
            if ppm_shift is not None:
                # we've defined the ppm shift
                self.update_b0_shift(ppm_shift)
            else:
                reference_peaks = []
                if len(self.metabolite_names):
                    if 'n-acetylaspartate' in self.metabolite_names:
                        reference_peaks.append(NAA_REFERENCE)
                    if 'creatine' in self.metabolite_names:
                        reference_peaks.append(CR_REFERENCE)
                else:
                    if not reference_peaks and ppm_shift is None:
                        # no ref metabolites found, try them anyway.
                        reference_peaks = [NAA_REFERENCE, CR_REFERENCE]

                if reference_peaks:
                    fft = self.fft()
                    nu = self.nu()
                    for reference_signal in reference_peaks:
                        peak = self.get_peak_location(fft, nu, reference_signal, 0.25)
                        if peak:
                            self.update_b0_shift(self._b0_ppm_shift + (peak - reference_signal))
                            break

    def clear_b0(self):
        self._correct_b0 = False
        self._b0_ppm_shift = 0
        self._b0_fft_pts_shift = 0
        self._b0_nu_shift = 0
        self._need_fft_cache_update = True

    def update_b0_shift(self, new_b0_shift):
        self._b0_ppm_shift = new_b0_shift
        self._b0_fft_pts_shift = self.ppm_to_nu_pts(self._b0_ppm_shift)
        self._b0_nu_shift = self._b0_ppm_shift - self.nu_pts_to_ppm(self._b0_fft_pts_shift)
        self._need_fft_cache_update = True

    def raw_adc(self):
        if self.scale == 0:
            return np.zeros(len(self._raw_adc), dtype=complex)
        else:
            return self._raw_adc * self.scale

    def trim_fft(self, high_ppm=-4.5, low_ppm=-1):
        nu = self.nu()
        if (np.max(nu) < low_ppm) or (np.min(nu) > high_ppm):
            raise Exception('Requested ppm trim range out of range of nu of spectra.')
        index = (nu > high_ppm) & (nu < low_ppm)
        return self.fft()[index], nu[index]

    def rescale_fft(self, high_ppm=-4.5, low_ppm=-1, npts=2048):
        # zero pads the time domain to fill the desired window with npts
        recursion_limit = 500
        nu = self.nu()
        if (np.max(nu) < low_ppm) or (np.min(nu) > high_ppm):
            raise Exception('Requested ppm rescale range out of range of nu of spectra. Max:' + str(np.min(nu)) +
                            ' Min: ' + str(np.max(nu)))

        # calculate initially how many points in that range
        index = (nu >= high_ppm) & (nu <= low_ppm)
        nu_pts = len(nu[index])
        counter = 0
        while nu_pts != npts:
            if counter > recursion_limit:
                raise Exception('Recursion limit hit!')
            if counter > 100:
                print('Counter is getting high... ' + str(self._zero_pad) + ' : ' + str(nu_pts) + ' aiming for: ' + str(
                    npts))

            # fine tune if that's not quite right
            # nu_pts = len(self.nu()[(self.nu() > high_ppm) & (self.nu() < low_ppm)])
            percent_range = len(nu) / float(nu_pts)
            # random is added as there is sometimes some aliasing effects, this corrects it
            self.update_zero_pad(
                self._zero_pad + int(round(np.round((npts - nu_pts) * percent_range) + np.random.random())))

            # print(str(self._zero_pad) + ' : ' + str(nu_pts) + ' aiming for: ' + str(npts) + ' dt:' + str(self.dt) + ' n:' + str(len(self._raw_adc)))

            if self._zero_pad < 0:
                raise Exception('Real data is too large to be input into the network, would have to reduce the '
                                'resolution of it. OR train a network with a higher resolution across the ppm range.')
            nu = self.nu()
            index = (nu >= high_ppm) & (nu <= low_ppm)
            nu_pts = len(nu[index])
            counter += 1

        return self.fft(update_cache=True)[index], nu[index]
