import os
import subprocess

import sys

sys.path.append('..')
from helpers.settings import slicer_folder


def resample_image(input_volume, reference_volume, output_volume, output_type, interpolation, warp_transform=None):
    cmd = list()
    cmd.append(os.path.join(slicer_folder, 'Slicer'))
    cmd.append('--launch')
    cmd.append(os.path.join(slicer_folder, 'lib', 'Slicer-4.10', 'cli-modules', 'BRAINSResample'))
    cmd.append('--inputVolume')
    cmd.append(input_volume)
    cmd.append('--referenceVolume')
    cmd.append(reference_volume)
    cmd.append('--outputVolume')
    cmd.append(output_volume)
    if warp_transform is not None:
        cmd.append('--warpTransform')
        cmd.append(warp_transform)
    cmd.append('--pixelType')
    # options: float, short, ushort, int, uint, uchar, binary
    cmd.append(output_type)
    cmd.append('--interpolationMode')
    # options: NearestNeighbor, Linear
    cmd.append(interpolation)
    cmd.append('--defaultValue')
    cmd.append('0')
    cmd.append('--numberOfThreads')
    cmd.append('-12')
    print(cmd)
    subprocess.call(cmd)

import glob
import os
import pathlib

ln2Path = "/mnt/data/LNQ-TIMC-totalseg-ln2/"
imagesTr = "/mnt/data/LNQ-TIMC/imagesTr"
imagesTs = "/mnt/data/LNQ-TIMC/imagesTs"

ln2s = glob.glob(ln2Path+"/*.nii.gz")
images = glob.glob(imagesTr+"/*.nii.gz")
images.extend(glob.glob(imagesTs+"/*.nii.gz"))

resultsPath = "/mnt/data/results/tcia-ln10_timc-2022-11-11"

imagesByName = {}
for image in images:
    imagesByName[os.path.basename(image)] = image

ln2sByName = {}
for ln2 in ln2s:
    name = os.path.basename(ln2)
    if name in imagesByName:
        ln2sByName[name] = ln2
    else:
        print(f"No image for {name}")


# for name in [list(ln2sByName.keys())[12]]:
counter = 0
for name in ln2sByName.keys():
    counter +=1
    resultPNGPath = f"{resultsPath}/{name}.png"
    if qt.QPixmap(resultPNGPath).size().width() != 0:
        print(f"skipping {resultPNGPath}")
        continue
    print(f"Inference on {counter} of {len(ln2sByName.keys())}")
    slicer.mrmlScene.Clear()
    slicer.app.processEvents()
    slicer.util.loadVolume(imagesByName[name])
    slicer.util.loadSegmentation(ln2sByName[name])
    slicer.util.selectModule("MONAILabel")
    slicer.app.processEvents()
    slicer.modules.MONAILabelWidget.onClickSegmentation()
    slicer.app.processEvents()
    seg = slicer.util.getNode("segmentation*")
    slicer.util.saveNode(seg, f"{resultsPath}/{name}.seg.nrrd")
    qt.QPixmap.grabWidget(slicer.util.mainWindow()).save(resultPNGPath)
    slicer.util.delayDisplay(f"{counter}")


screenshots = glob.glob(f"{resultsPath}/*.png")

reviewWindow = qt.QWidget()
reviewLayout = qt.QVBoxLayout()
reviewWindow.setLayout(reviewLayout)

reviewSlider = ctk.ctkSliderWidget()
reviewSlider.decimals = 0
reviewSlider.maximum = len(screenshots) -1
reviewLayout.addWidget(reviewSlider)

reviewButton = qt.QPushButton("Load")
reviewLayout.addWidget(reviewButton)

reviewImage = qt.QLabel()
reviewLayout.addWidget(reviewImage)
reviewImage.setPixmap(qt.QPixmap(screenshots[0]))

reviewSlider.connect("valueChanged(double)", lambda value: reviewImage.setPixmap(qt.QPixmap(screenshots[int(value)])))

def loadCase():
    slicer.mrmlScene.Clear()
    screenShotPath = screenshots[int(reviewSlider.value)]
    name = os.path.basename(screenShotPath)[:-4]
    slicer.util.loadVolume(imagesByName[name])
    slicer.util.loadSegmentation(ln2sByName[name])
    slicer.util.loadSegmentation(f"{resultsPath}/{name}.seg.nrrd")
    slicer.util.mainWindow().raise_()

reviewButton.connect("clicked()", loadCase)

reviewWindow.show()

