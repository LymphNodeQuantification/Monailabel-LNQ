import os
import sys
from collections import OrderedDict

import SimpleITK as sitk
import pandas as pd
import pydicom
from tqdm import tqdm

sys.path.append('../..')
from helpers.settings import raw_folder, images_folder, sheets_folder
from helpers.utils import read_dicom_gdcm

if __name__ == '__main__':
    ct_lymph_nodes_folder = os.path.join(raw_folder, 'tcia_ct_meds')
    output_image_folder = os.path.join(images_folder, 'tcia_ct_lymph_nodes')
    if not os.path.isdir(output_image_folder):
        os.mkdir(output_image_folder)
    #
    masks_folder = os.path.join(raw_folder, 'MED_ABD_LYMPH_MASKS')

    patient_folders = sorted(os.listdir(ct_lymph_nodes_folder))

    patient_info = list()
    images_info = list()

    for index, patient_folder in enumerate(tqdm(patient_folders)):
        study_folders = os.listdir(os.path.join(ct_lymph_nodes_folder, patient_folder))
        assert len(study_folders) == 1
        series_folders = os.listdir(os.path.join(ct_lymph_nodes_folder, patient_folder, study_folders[0]))
        assert len(series_folders) == 1
        series_folder = os.path.join(ct_lymph_nodes_folder, patient_folder, study_folders[0], series_folders[0])
        image_paths = sorted(os.listdir(series_folder))
        sample_image = image_paths[0]
        ds = pydicom.dcmread(os.path.join(series_folder, sample_image))
        image = read_dicom_gdcm(series_folder)
        image_size = image.GetSize()
        image_spacing = image.GetSpacing()
        patient_info.append(OrderedDict(
            {
                "case": patient_folder,
                "gender": ds.PatientSex,
                "age": ds.PatientAge
            }
        ))
        images_info.append(OrderedDict(
            {
                "case": patient_folder,
                "spacing i": image_spacing[0],
                "spacing j": image_spacing[1],
                "spacing k": image_spacing[2],
                "size i": image_size[0],
                "size j": image_size[1],
                "size k":  image_size[2]
            }
        ))
        output_image_path = os.path.join(output_image_folder, patient_folder + '.nrrd')
        sitk.WriteImage(image, output_image_path, True)
        mask_path = os.path.join(masks_folder, patient_folder, patient_folder + '_mask.nii.gz')
        mask = sitk.ReadImage(mask_path)
        output_mask_path = os.path.join(output_image_folder, patient_folder + '_label.nrrd')
        sitk.WriteImage(mask, output_mask_path, True)

    pd.DataFrame(patient_info).to_csv(os.path.join(sheets_folder, 'patient_info.csv'), index=False)
    pd.DataFrame(images_info).to_csv(os.path.join(sheets_folder, 'images_info.csv'), index=False)
