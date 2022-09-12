import glob
import os
import sys

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from tqdm import tqdm

sys.path.append('..')
from helpers.settings import images_folder
from helpers.utils import resample

pad = 5

if __name__ == '__main__':
    ct_folder = os.path.join(images_folder, 'tcia_ct_lymph_nodes')
    lungs_folder = os.path.join(images_folder, 'lungs')

    resampled_output_folder = os.path.join(images_folder, 'preprocessed')
    if not os.path.isdir(resampled_output_folder):
        os.mkdir(resampled_output_folder)

    ln_paths = sorted(glob.glob(ct_folder + '/*_label.nrrd'))

    for ln_path in tqdm(ln_paths):
        # '''
        ct_path = ln_path.replace('_label', '')
        lung_path = os.path.join(lungs_folder, os.path.basename(ln_path))

        ct_image = sitk.ReadImage(ct_path)
        ln_image = sitk.ReadImage(ln_path)
        lung_image = sitk.ReadImage(lung_path)
        lung_nda = sitk.GetArrayFromImage(lung_image)
        lung_sums = np.sum(lung_nda, axis=(1, 2))
        cut_index = max(0, int(np.where(lung_sums > 0)[0][0] - pad))
        print(ct_path, cut_index)
        cropped_ct = sitk.Crop(ct_image, [0, 0, cut_index], [0, 0, 0])
        cropped_ln = sitk.Crop(ln_image, [0, 0, cut_index], [0, 0, 0])
        cropped_ln = sitk.BinaryThreshold(cropped_ln,
                                              lowerThreshold=1,
                                              upperThreshold=255,
                                              insideValue=1, outsideValue=0)

        # write to output
        sitk.WriteImage(cropped_ct, os.path.join(resampled_output_folder, os.path.basename(ct_path)))
        sitk.WriteImage(cropped_ln, os.path.join(resampled_output_folder, os.path.basename(ln_path)), True)
