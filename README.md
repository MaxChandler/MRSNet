# MRSNet

MR spectral quantification using convoloutional neural networks.

This framework provides methods to generate datasets from loaded LCModel ".BASIS" files or simulated by [FID-A](https://github.com/CIC-methods/FID-A) or 
[PyGamma](https://scion.duhs.duke.edu/vespa/gamma/wiki/PyGamma). 

## Built With
### Software
* [Keras](https://keras.io/) - The Deep Learning framework used
* [Tensorflow](https://www.tensorflow.org/) - Underlying Machine Learning library
* [FID-A](https://github.com/CIC-methods/FID-A) - MRS simulation toolbox
* [PyGamma](https://scion.duhs.duke.edu/vespa/gamma/wiki/PyGamma) - Another MRS simulation toolbox
* [VeSPA](https://scion.duhs.duke.edu/vespa/project) - Versatile Simulation, Pulses and Analysis 
### Data sources
* [Swansea Benchmark Dataset](https://langbein.org/gabaphantoms_20190815) - Benchmark phantom datasets collected at Swansea University's 3T Siemens scanner.
* [Purdue LCModel basis sets](http://purcell.healthsciences.purdue.edu/mrslab/basis_sets.html) - Data source for the LCModel basis sets

## Getting Started
### Prerequisites
* Python 2.7 - Unfortunately, PyGamma still requires Python 2.7 to run so the entire library is written in Python 2.7
* MATLAB - Only required if you plan to simulate new FID-A spectra. 
* Linux system packages: 
    * Git-lfs for git submodule support: `git-lfs` 
    * Python2 virtual environment to isolate installation packages and environment: `python-virtualenv`
* Install these using your package manager with root privileges. E.g. Debian based distributions: `sudo apt update && sudo apt install git-lfs python-virtualenv`

### Install instructions (Linux)

For simplicity, python packages are installed into a virtual environment. 
1. Clone the repository: `git clone https://qyber.black/MRIS/mrsnet.git`
2. Navigate to the directory: `cd mrsnet`
3. Update submodules: `git submodule init && git submodule update`
4. Create a virtual environment 'venv': `python2 -m virtualenv venv`
5. Activate the virtual environment: `source venv/bin/activate`
6. Install the requirements (CPU or GPU):
    1. CPU version: `pip install -r requirements.txt`
    2. GPU version (requires [CUDA](https://developer.nvidia.com/cuda-zone) : There's a good guide available [here](https://www.tensorflow.org/install/gpu)) : `pip install -r gpu_requirements.txt`
7. Download the additional required data: `python2 setup.py`


## Training a network

Training a network is the default for model.py:

```
python2 model.py
```
To see a list of options and their defaults, call:
```
python2 model.py --help
```
An example of a more complex training call:
```
python2 model.py -N 10000 -e 200 -b 16 -d fida --linewidths 0.75 1 1.25 --omega 900 --model_name mrsnet_small_kernel_no_pool
```
This above example will simulate spectra and train a network with:
* Train for 200 epochs, with a mini-batch size of 16
* Spectra are sourced from FID-A, with a scanner B0 field of 900MHz
* 10,000 Spectra are evenly split (3,333) over the linewidths (0.75, 1, 1.25)
* For the network architecture called "mrsnet_small_kernel_no_pool" (found in [networks.py](networks.py))

#### Network performance
By default, networks are stored in `MRSNet/models/`along with some basic analytics.

## Quantifying Spectra
Quantifying spectra:
```python2
python2 model.py -m quantify
```
Defaults are to use the E1 MEGA-PRESS benchmark spectra, with the best network from MRSNet. Output will 


Quantifying spectra and specifying the model and spectra directory:
```
python2 model.py -m quantify --network_path "models/complete/some_model_dir/" --spectra_path "some/spectra/directory/"
```
The default behaviour of quantify is to use the best network from the MRSNet paper to quantify the bundled E3 dataset:
```
Quantifying 13 MEGA-PRESS Spectra
This network can only quantify: ['creatine', 'gaba', 'glutamate', 'glutamine', 'n-acetylaspartate']
	Network path: ./models/complete/MRSNet/
	DICOM path: ./Datasets/Benchmark/E1/MEGA-PRESS/1250Hz/


Spectra ID: GABA_SERIES_00P5MM
	Metabolite           Quantity
	Creatine             0.077157
	GABA                 0.556846
	Glutamate            0.114418
	Glutamine            0.203197
	N-Acetylaspartate    0.048382
```

### Quantifying your own MEGA-PRESS spectra
The code will attempt to analyse all of the spectra contained in the provided directory. 
There are a couple of caveats to enable this to work correctly:

1. All three acquisitions for each MEGA-PRESS scan must be present (edit on, edit off, difference).
2. Spectra that belong to the same scan must have a unique ID of your choise added to their filename (e.g. SCAN_001).
	1. If you know the ground truth of the scan, it should be added to the dictionary located in Utilities/constants.py. This enables the code to know which spectra are from the same scans to group them together.
3. Spectra of the different acquisition types must be labeled, by adding either "EDIT_OFF", "EDIT_ON" or "DIFF" to anywhere after the unique ID from 2 in their filename.

An example for two MEGA-PRESS scan would be six files: 
```
"SCAN_000_EDIT_OFF.ima"
"SCAN_000_EDIT_ON.ima"
"SCAN_000_DIFF.ima"
"SCAN_001_EDIT_OFF.ima"
"SCAN_001_EDIT_ON.ima"
"SCAN_001_DIFF.ima"
```

### Non-Siemens DICOM files
Loading of non-Siemens DICOM files has not been tested.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 


## License

This project is licensed under the GNP [AGPLv3 License](https://choosealicense.com/licenses/agpl-3.0/), see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Brian Soher (VeSPA/PyGamma) for help locating the PyGamma pulse sequence code for MEGA-PRESS, PRESS and STEAM.

