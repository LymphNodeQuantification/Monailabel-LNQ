import glob
import os
import sys

import SimpleITK as sitk

sys.path.append('../..')
from helpers.settings import images_folder
from helpers.utils import segment_lung

if __name__ == '__main__':
    input_images_folder = os.path.join(images_folder, 'tcia_ct_lymph_nodes')
    input_body_folder = os.path.join(images_folder, 'bodies')
    output_folder = os.path.join(images_folder, 'lungs')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    label_paths = sorted(glob.glob(input_images_folder + '/*_label.nrrd'))
    for label_path in label_paths:
        ct_path = label_path.replace('_label', '')
        body_path = os.path.join(input_body_folder, os.path.basename(label_path))
        #
        ct_image = sitk.ReadImage(ct_path)
        body_image = sitk.ReadImage(body_path)
        ct_image = sitk.Mask(ct_image, body_image)
        #
        lung_image = segment_lung(ct_image)
        sitk.WriteImage(lung_image, os.path.join(output_folder, os.path.basename(label_path)), True)
