"""

Helper script to add lymph node labels from the TCIA collection to the labels from TotalSegmentator to create the ln, ln2, and ln10 datasets.

This was run in Slicer 5.1-2022-20-20 (patched since the 5.0.3 release to correctly read nifti files from TotalSegmentator).

exec(open("/mnt/vessels/tcia/tcia-lnq-totalseg.py").read())

"""

import glob
import numpy
import os

# inputs
tciaPath = "/mnt/vessels/tcia/tcia_lymphnode_monailabel"
tciaLNOnlyPath = "/mnt/vessels/tcia/tcia_lymphnode_monailabel/labels/final"
totalSegPath = "/mnt/vessels/tcia/segmentations_preprocessed-v1.3"

# outputs
lnqTCIAPath = "/mnt/vessels/tcia/tcia-totalseg-ln"
ln2TCIAPath = "/mnt/vessels/tcia/tcia-totalseg-ln2"
ln10TCIAPath = "/mnt/vessels/tcia/tcia-totalseg-ln10"

lnOnlyCases = glob.glob(f"{tciaLNOnlyPath}/*.nrrd")

def getCaseID(synthSegPath):
    return synthSegPath.split("/")[-1][:-1*len(".nrrd")]

for lnOnlyCase in lnOnlyCases:
    print(lnOnlyCase)
    slicer.mrmlScene.Clear()
    caseID = getCaseID(lnOnlyCase)
    totalsegLNPath = f"{lnqTCIAPath}/{caseID}.nii.gz"
    totalsegLN2Path = f"{ln2TCIAPath}/{caseID}.nii.gz"
    totalsegLN10Path = f"{ln10TCIAPath}/{caseID}.nii.gz"
    slicer.util.delayDisplay(f"Starting case {caseID}", 100)
    if os.path.exists(totalsegLNPath):
        print(f"Skipping case {caseID}", 100)
        continue

    ctPath = f"{tciaPath}/{caseID}.nrrd"
    lnPath = f"{tciaLNOnlyPath}/{caseID}.nrrd"
    tsPath = f"{totalSegPath}/{caseID}.nii.gz"
    ctNode = slicer.util.loadVolume(ctPath)
    lnLabel = slicer.util.loadLabelVolume(lnPath)
    lnLabel.SetName("lnLabel")
    tsLabel = slicer.util.loadLabelVolume(tsPath)
    tsLabel.SetName("tsLabel")

    slicer.app.processEvents(qt.QEventLoop.ExcludeUserInputEvents)

    lnArray = slicer.util.arrayFromVolume(lnLabel)
    tsArray = slicer.util.arrayFromVolume(tsLabel)

    lnCoords = numpy.where(lnArray == 1)
    print(tsArray[lnCoords].mean())
    tsArray[lnCoords] = 255
    print(tsArray[lnCoords].mean())

    # ln2
    volumesLogic = slicer.modules.volumes.logic()
    ln2Label = volumesLogic.CloneVolume(tsLabel, "ln2Label")
    ln2Array = slicer.util.arrayFromVolume(ln2Label)

    tsCoords = numpy.where(tsArray != 0)
    ln2Array[tsCoords] = 1
    ln2Array[lnCoords] = 2

    # ln10
    volumesLogic = slicer.modules.volumes.logic()
    ln10Label = volumesLogic.CloneVolume(tsLabel, "ln10Label")
    ln10Array = slicer.util.arrayFromVolume(ln10Label)

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
    for segmentMap in segmentMaps:
        start, end, mapTo = segmentMap
        for tsValue in range(start,end+1):
            tsCoords = numpy.where(tsArray == tsValue)
            ln10Array[tsCoords] = mapTo
    ln10Array[lnCoords] = 11

    slicer.util.arrayFromVolumeModified(tsLabel)
    slicer.util.arrayFromVolumeModified(ln2Label)
    slicer.util.arrayFromVolumeModified(ln10Label)
    slicer.util.saveNode(tsLabel, totalsegLNPath)
    slicer.util.saveNode(ln2Label, totalsegLN2Path)
    slicer.util.saveNode(ln10Label, totalsegLN10Path)

