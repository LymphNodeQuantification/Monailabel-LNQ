# LymphNodeMonailabel

CT data for training this model is available [on dropbox here in nrrd format](https://www.dropbox.com/s/cb7gpkeou25x3zk/tcia_lymphnode_monailabel%20copy.zip?dl=0).

It is originally from the [TCIA CT Lymph Nodes collection](https://wiki.cancerimagingarchive.net/display/Public/CT+Lymph+Nodes) in the orginal DICOM.

After downloading the server can be started like this (assumes MONAI Label is already installed and working in your python environment).

```

monailabel start_server --app apps/radiology --studies datasets/tcia-ln10/imagesTr --conf models segmentation

```

Note: this is a work-in-progress model that does not yet segment lymph nodes correctly.

# Preprocessed segmentations

This link contains three versions of the data run through [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) and then combined with the manual segmentations from the CT Lymph Node collecion:

https://www.dropbox.com/sh/ish3ci1h9zkbbhu/AAC84tfHmQ5whW0HF-bqe5vGa?dl=0

The three subdirectories contain:
* *tcia-totalseg-ln* The [full]([url](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py#L2-L106)) TotalSegmentator segmentations with the Lymph Node segment added as label 255
* *tcia-totalseg-ln2* All TotalSegmentator segments are collapsed to the label value 1, and lymph nodes are label value 2
* *tcia-totalseg-ln10* TotalSegmentator classes reduced to 10 label values according to the mapping below and lymph nodes are label 11

Mapping for ln10 data:
```
    segmentMaps = [
            [1, 12, 1], # organs
            [13, 17, 2], # lungs
            [18, 41, 3], # spine
            [42, 49, 4], # cardio-pulmonary
            [50, 50, 5], # brain
            [51, 57, 6], # guts
            [58, 87, 7], # bones
            [88, 92, 8], # hips
            [93, 93, 9], # face
            [94, 103, 10], # muscles
            [104, 104, 6], # bladder
    ]
```
All segmentations are in nii.gz format.

## Related discussions

* [Suggestions from the MONAI Label team](https://github.com/Project-MONAI/MONAILabel/discussions/1026#discussioncomment-3771449).


# Acknowledgements

This work is supported by the US National Institutes of Health, National Cancer Institutes grant [5R01CA235589 Lymph Node Quantification System for Multisite Clinical Trials
](https://reporter.nih.gov/search/9w2EcX30REWk0JwFTROtVw/project-details/10479822).  Additional support provided by the [NCI Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/).  
