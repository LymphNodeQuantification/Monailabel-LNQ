"""

Helper script to add lymph node labels from the from TIMC diamonds to the labels from TotalSegmentator to create the ln, ln2, and ln10 datasets.

This was run in Slicer 5.2.1

exec(open("/mnt/lnq-timc-batch1/lnq-timc-ts-ln10.py").read())

Results are in gs://lnq-timc-batch1-seg1/ts-ln

"""

import glob
import numpy
import os

# inputs
timcPath = "/mnt/lnq-timc-batch1/ct-series"
timcDiamondsPath = "/mnt/lnq-timc-batch1/measurements-segmentations-.5"
totalSegPath = "/mnt/lnq-timc-batch1/ct-ts-segmentations"

# outputs
lnqTCIAPath = "/mnt/lnq-timc-batch1/ct-series-ln-.5"
ln2TCIAPath = "/mnt/lnq-timc-batch1/ct-series-ln2-.5"
ln10TCIAPath = "/mnt/lnq-timc-batch1/ct-series-ln10-.5"

lnOnlyCases = glob.glob(f"{timcDiamondsPath}/*.nii.gz")

casesToRerun = []
for lnOnlyCase in lnOnlyCases:
    print(lnOnlyCase)
    caseID = lnOnlyCase.split("/")[-1][:-1*len(".nii.gz")]
    totalsegLNPath = f"{lnqTCIAPath}/{caseID}.nii.gz"
    totalsegLN2Path = f"{ln2TCIAPath}/{caseID}.nii.gz"
    totalsegLN10Path = f"{ln10TCIAPath}/{caseID}.nii.gz"
    slicer.util.delayDisplay(f"Starting case {caseID}", 100)
    if os.path.exists(totalsegLNPath):
        print(f"Skipping case {caseID}", 100)
        continue

    slicer.mrmlScene.Clear()
    ctPath = f"{timcPath}/{caseID}.nii.gz"
    lnPath = f"{timcDiamondsPath}/{caseID}.nii.gz"
    tsPath = f"{totalSegPath}/{caseID}.nii.gz"
    ctNode = slicer.util.loadVolume(ctPath)
    lnLabel = slicer.util.loadLabelVolume(lnPath)
    lnLabel.SetName("lnLabel")
    try:
        tsLabel = slicer.util.loadLabelVolume(tsPath)
    except RuntimeError:
        print(f"could not load {tsPath}")
        casesToRerun.append(caseID)
        continue
    tsLabel.SetName("tsLabel")

    slicer.app.processEvents(qt.QEventLoop.ExcludeUserInputEvents)

    lnArray = slicer.util.arrayFromVolume(lnLabel)
    tsArray = slicer.util.arrayFromVolume(tsLabel)

    lnCoords = numpy.where(lnArray != 0)
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

print(f"Finished.  Reruns:")
print(casesToRerun)
