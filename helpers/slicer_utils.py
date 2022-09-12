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
