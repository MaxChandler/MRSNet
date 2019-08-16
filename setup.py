#!/usr/bin/env python2.7
import urllib2
import os
import zipfile
from tqdm import tqdm
import requests

def main():
    # Download LCModel basis sets from http://purcell.healthsciences.purdue.edu/mrslab/basis_sets.html
    save_directory = './Basis/simulated/LCModel/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    urls = ["http://purcell.healthsciences.purdue.edu/mrslab/files/3t_IU_MP_te68_748_ppm_inv.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_IU_MP_te68_diff_yesNAAG_noLac_Kaiser.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_IU_MP_te68_diff_yesNAAG_noLac_c.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_GE_MEGAPRESS_Kaiser_oct2011_75_ppm_inv.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_GE_MEGAPRESS_Kaiser_oct2011_1975_diff.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_GE_MEGAPRESS_june2011_diff.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_philips_MEGAPRESS_Kaiser_oct2011_75_ppm_inv.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_philips_MEGAPRESS_Kaiser_oct2011_1975_diff.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_philips_MEGAPRESS_may2010_diff.basis"]

    filenames = ["MEGAPRESS_edit_off_Siemens_3T.basis",
                 "MEGAPRESS_difference_Siemens_3T_kasier.basis",
                 "MEGAPRESS_difference_Siemens_3T_govindaraju.basis",
                 "MEGAPRESS_edit_off_GE_3T.basis",
                 "MEGAPRESS_difference_GE_3T_kasier.basis",
                 "MEGAPRESS_difference_GE_3T_govindaraju.basis",
                 "MEGAPRESS_edit_off_Phillips_3T.basis",
                 "MEGAPRESS_difference_Phillips_3T_kasier.basis",
                 "MEGAPRESS_difference_Phillips_3T_govindaraju.basis"]

    for url, filename in tqdm(zip(urls, filenames), total=len(urls), desc='Downloading LCModel basis sets'):
        response = urllib2.urlopen(url)
        with open(save_directory + filename, 'wb') as output:
            output.write(response.read())

    print('Finished downloading LCModel MEGA-PRESS basis sets from Purdue for Siemens, Phillips & GE.')

    # from here we assume the .zip file is in the directory
    if not os.path.exists('./Datasets/Benchmark/'):
        os.makedirs('./Datasets/Benchmark/')

    print('Downloading experimental benchmark datasets. Warning, this file is large < 3GB.')
    zip_filename = download_large_file('https://qyber.black/data/MRIS/phantoms/GABAPhantoms_20190815.zip')
    print('Need to add download URL')

    print('Extracting experimental benchmark datasets to ./Datasets/Benchmark/')
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('./Datasets/Benchmark/')

    print('Deleting downloaded .zip file')
    os.remove(zip_filename)
    print('Done!')

def download_large_file(url):
    # https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        filesize = int(requests.get(url, stream=True).headers['Content-length'])
        chunk_size = 8192
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=filesize/chunk_size, desc='Downloading experimental benchmark datasets'):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename

if __name__ == '__main__':
    main()
