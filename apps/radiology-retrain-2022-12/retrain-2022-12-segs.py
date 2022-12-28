
import glob
import os

correctionsDir="/mnt/sdp-timc-correction/timc-ln10-2022-11-11/segmentations-corrected"
correctionsNiiDir="/mnt/sdp-timc-correction/timc-ln10-2022-11-11/segmentations-corrected-nii"

count = 0
nrrds = glob.glob(f"{correctionsDir}/*.nrrd")
for nrrd in nrrds:
    count += 1
    print(f"Loading... {count} of {len(nrrds)}")
    print(nrrd)
    basename = os.path.basename(nrrd)
    print(basename)
    name = basename.replace("timc-ln10-2022-11-11_segmentations_", "")
    name = name.replace(" (1)", "")
    name = name.replace(".seg.nrrd", "")
    print(name)
    niiPath = f"{correctionsNiiDir}/{name}.nii.gz"
    volume = slicer.util.loadVolume(nrrd)
    print("Saving...")
    slicer.util.saveNode(volume,niiPath)
    print("Clearing...")
    slicer.mrmlScene.Clear()
    slicer.app.processEvents()
    slicer.util.delayDisplay(f"done with {niiPath}")


