from __future__ import division

import os
import pdb
import numpy as np
import dill
from pygamma_pulse_sequences import fid, press, steam, megapress
from datetime import datetime
from Utilities.constants import OMEGA, FULL_METABOLITE_LIST
import Basis
from Spectra import Spectra


def cache(save_dir, spectra):
    for s in spectra:
        s._dss = None  # no need to save an additional dss object if we don't have too..
        s.check()
    with open(os.path.join(save_dir, (spectra[0].metabolite_names[0]) + '.dill'), 'wb') as save_file:
        dill.dump(spectra, save_file)


def pygamma_spectra(metabolite_names, pulse_sequence='megapress', omega=OMEGA, linewidth=1.0,
                    npts=4096, adc_dt=4e-4, concentrations=None, testing=False):
    # by having multiple linewidths it allows the use of the same 'mx' object to run the binning over,
    # rather than have to recalcualte the pulse sequence again. it would be more efficient to save the mx table
    # but this functionality is currently (Sept-2018) broken in PyGamma
    import pygamma as pg
    linewidth = float(linewidth)
    to_sim = list(metabolite_names)

    basis = Basis.Basis()
    for metabolite_name in metabolite_names:
        if not testing:
            cache_name = pulse_sequence + '_' + str(omega) + '_' + str(linewidth) + '_' + str(npts) + '_' + str(adc_dt)
            spectra_cache_dir = './cache/pygamma/' + cache_name + '/'
            if os.path.isdir(spectra_cache_dir):
                for file in os.listdir(spectra_cache_dir):
                    if file == metabolite_name + '.dill':
                        # print('Cache hit, already simulated '+pulse_sequence+' for ' + metabolite_name +'.')
                        with open(os.path.join(spectra_cache_dir, file), 'rb') as load_file:
                            for spectra in dill.load(load_file):
                                basis.add_spectra(spectra)
                        to_sim.remove(metabolite_name)
            else:
                os.makedirs(spectra_cache_dir)

    for metabolite_name in to_sim:
        id = hash(datetime.now().strftime('%f%S%H%M%a%d%b%Y'))
        mol_spectra = []
        print('Cache miss. Simulating ' + metabolite_name + ' : ' + pulse_sequence)
        infile = os.path.join('./Simulators/PyGamma/metabolite models', metabolite_name.lower() + '.sys')
        spin_system = pg.spin_system()
        spin_system.read(infile)
        spin_system.OmegaAdjust(omega)

        if pulse_sequence.lower() == 'fid':
            mx = [fid(spin_system)]
        elif pulse_sequence.lower() == 'press':
            mx = [press(spin_system)]
        elif pulse_sequence.lower() == 'steam':
            mx = [steam(spin_system)]
        elif pulse_sequence.lower() == 'megapress':
            mx = megapress(spin_system, omega=omega)
        else:
            raise Exception('Unrecognised PyGamma pulse sequence: ' + pulse_sequence)

        for count, acq in enumerate(mx):
            spectra = Spectra()
            spectra.source = 'pygamma'
            spectra.type = 'simulated'
            spectra.id = id
            spectra.pulse_sequence = pulse_sequence.lower()
            spectra.metabolite_names = [metabolite_name]
            spectra.concentrations = [1]
            spectra.omega = omega
            spectra.npts = npts
            spectra.dt = adc_dt
            spectra.linewidth = linewidth
            spectra._raw_adc = binning(acq, linewidth=linewidth, dt=adc_dt, npts=npts)
            spectra.pulse_sequence = pulse_sequence
            spectra.center_ppm = 0
            spectra.acquisition = count

            if pulse_sequence.lower() == 'megapress':
                if count == 0:
                    spectra.spectra_type = 'edit off'
                elif count == 1:
                    spectra.spectra_type = 'edit on'
                else:
                    raise Exception('More than 2 mx objects for megapress? Something is wrong here...')

            basis.add_spectra(spectra)
            mol_spectra.append(spectra)

        # and force gc to try and delete the C++ PyGamma mx object
        for acq in mx:
            acq.disown()
            del acq
        del mx

        # finally we cache the spectra
        if not testing:
            cache(spectra_cache_dir, mol_spectra)

    basis.setup()
    if pulse_sequence.lower() == 'megapress':
        basis.add_difference_spectra()
    basis.update_spectra_scale(concentrations=concentrations)
    return basis


def binning(mx, linewidth=1, dt=5e-4, npts=2048):
    # add some broadening and decay to the model
    mx.resolution(0.5)              # Combine transitions within 0.5 rad/sec
    mx.broaden(linewidth)
    acq = mx.T(npts, dt)
    ADC = []
    for ii in range(0, acq.pts()):
        ADC.extend([acq.get(ii).real() + (1j * acq.get(ii).imag())])
    return np.array(ADC)


if __name__ == '__main__':
    linewidths = [0.5, 0.75, 1, 1.25, 1.5]
    for pulse_sequence in ['megapress']: #'fid', 'press', 'steam',
        for linewidth in linewidths:
            pygamma_spectra(FULL_METABOLITE_LIST, pulse_sequence=pulse_sequence, omega=OMEGA, linewidth=linewidth)
