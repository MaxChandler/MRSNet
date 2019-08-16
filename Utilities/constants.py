#!/usr/bin/env python2.7
# ORDER MUST STAY : these are already sorted, but here is a reminder here that they are sorted.
FULL_METABOLITE_LIST = sorted(['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate'])
OMEGA = 123.23                      # 2.89T for the Swansea University scanner
NAA_REFERENCE = -2.01
CR_REFERENCE = -3.015
WATER_REFERENCE = -4.75             # Temperature dependant, avoid using if at all possible
GYROMAGNETIC_RATIO = 42.57747892    # 1H (MHz/T) : https://physics.nist.gov/cgi-bin/cuu/Value?gammapbar

"""
This is not the best idea for storing metadata about the scan data, but it works and is simple.

It attaches data to spectra loaded from DICOM sources where the dictionary key (below) is somewhere in the filename.
E.g for one of the E1 MEGA-PRESS phantom scans it has "GABA_SERIES_NAA_ONLY" in the file name. This ID acts as the
ID for the group of spectra, it is required for MEGA-PRESS spectra to correctly group them.

Additionally EDIT_ON, EDIT_OFF and DIFF are needed to identify what acqusition they are (see Spectra->load_dicom()
for more information).
"""

DICOM_METADATA ={
    'E1':{
        'GABA00_NAA_15mM_Cr_0mM': {'metabolite_names': ['n-acetylaspartate'], 'concentrations': [15.0]},
        'GABA00_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine'], 'concentrations': [15.0, 8.0]},
        'GABA01_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 0.5]},
        'GABA02_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 1.0]},
        'GABA03_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 1.5]},
        'GABA04_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 2.0]},
        'GABA05_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 2.5]},
        'GABA06_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 3.0]},
        'GABA07_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 4.0]},
        'GABA08_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 6.0]},
        'GABA09_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 8.0]},
        'GABA10_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 10.0]},
        'GABA11_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 11.6]}
    },
    'E2':{
        'GABA00_NAA_15mM_Cr_0mM': {'metabolite_names': ['n-acetylaspartate'], 'concentrations': [15.0]},
        'GABA00_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine'], 'concentrations': [15.0, 8.0]},
        'GABA01_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 0.5]},
        'GABA02_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 1.0]},
        'GABA03_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 1.5]},
        'GABA04_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 2.0]},
        'GABA05_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 2.5]},
        'GABA06_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 3.0]},
        'GABA07_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 4.0]},
        'GABA08_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 5.0]},
        'GABA09_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 6.0]},
        'GABA10_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 7.0]},
        'GABA11_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 8.0]},
        'GABA12_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 9.0]},
        'GABA13_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 10.0]},
        'GABA14_NAA_15mM_Cr_8mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba'], 'concentrations': [15.0, 8.0, 12.0]}
    },
    'E3':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 12, 3]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 12, 3]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1.99655172413793, 12, 3]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2.98966706302021, 12, 3]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3.97935786625118, 12, 3]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4.96563594257445, 12, 3]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 5.94851306001385, 12, 3]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6.9280009460138, 12, 3]},
        'GABA08_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 7.90411128757927, 12, 3]},
        'GABA09_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8.87685573141521, 12, 3]},
        'GABA10_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 9.8462458840655, 12, 3]},
        'GABA11_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10.8122933120515, 12, 3]},
        'GABA12_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 11.7750095420099, 12, 3]},
        'GABA13_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 12.7344060608306, 12, 3]},
        'GABA14_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 13.6904943157932, 12, 3]}
    },
    'E4a':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 12, 3]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 12, 3]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2, 12, 3]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 12, 3]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4, 12, 3]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6, 12, 3]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8, 12, 3]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10, 12, 3]}
    },
    'E4b':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 12, 3]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 12, 3]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2, 12, 3]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 12, 3]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4, 12, 3]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6, 12, 3]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8, 12, 3]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10, 12, 3]}
    },
    'E4c':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 12, 3]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 12, 3]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2, 12, 3]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 12, 3]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4, 12, 3]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6, 12, 3]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8, 12, 3]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10, 12, 3]}
    },
    'E4d':{
        'GABA00_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 12, 3]},
        'GABA01_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 1, 12, 3]},
        'GABA02_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 2, 12, 3]},
        'GABA03_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 3, 12, 3]},
        'GABA04_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 4, 12, 3]},
        'GABA05_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 6, 12, 3]},
        'GABA06_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 8, 12, 3]},
        'GABA07_NAA_15mM_Cr_8mM_Glu_12mM_Gln_3mM': {'metabolite_names': ['n-acetylaspartate', 'creatine', 'gaba', 'glutamine', 'glutamate'], 'concentrations': [15, 8, 10, 12, 3]}
    },
    # # Blank Example
    # 'Spectra_set_name':{
    #     'Some_Spectra_ID00': {},
    #     'Some_Spectra_ID01': {}
    # }
}
