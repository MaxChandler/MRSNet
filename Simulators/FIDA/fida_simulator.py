import subprocess
import os
from Utilities.constants import OMEGA, FULL_METABOLITE_LIST, GYROMAGNETIC_RATIO


def fida_spectra(metabolite_names, omega=OMEGA, linewidth=1.0, npts=4096, adc_dt=4e-4, save_dir='./cache/fida/'):
    matlab_command = "addpath(genpath([pwd,'/Simulators/FIDA/'])); "
    matlab_command += "Bfield=" + str(omega/GYROMAGNETIC_RATIO) + "; "
    matlab_command += "Npts=" + str(npts) + "; "
    matlab_command += "sw="+str(1/adc_dt)+"; "

    matlab_command += "metabolites={"
    for m in metabolite_names:
        matlab_command += '\'' + convert_metabolite_name(m) + '\','
    matlab_command = matlab_command.rstrip(',')
    matlab_command+="}; "

    matlab_command += "linewidths=[" + str(linewidth) + "]; "
    matlab_command += "save_dir='" + save_dir + "'; "

    matlab_command += "run_custom_simMegaPressShapedEdit; exit; exit;"

    try:
        p = subprocess.Popen(['matlab', '-nosplash', '-nodisplay', '-r', matlab_command])
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            raise Exception('Matlab is not installed on this system! Can\'t simulate FID-A spectra.\n'
                            'You can simulate them on another system, and put them into MRSNet/cache/fida/.')
        else:
            raise

    p.wait()


def convert_metabolite_name(name):
    # converts to the expected value for FID-A.
    # see FID-A/simulationTools/metabolites for options
    name = name.lower()
    m_names = {'creatine': 'Cr', 'gaba': 'GABA', 'glutamate': 'Glu', 'glutamine': 'Gln', 'lactate': 'Lac',
               'myo-inositol': 'Ins', 'n-acetylaspartate': 'NAA', 'taurine': 'Tau'}
    return m_names[name]
