import time

import SimpleITK as sitk
import numpy as np
from skimage import measure
from scipy import spatial
from tqdm import tqdm


def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S")


def read_dicom_gdcm(input_directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def segment_body(image):
    nda = sitk.GetArrayFromImage(image)
    nda[nda == nda[0, 0, 0]] = -1000
    binary_nda = np.array(nda > -400, dtype=np.int8) + 1

    for i, axial_slice in enumerate(tqdm(binary_nda)):
        axial_slice = axial_slice - 1
        labeling = measure.label(axial_slice)
        l_max = largest_label_volume(labeling, bg=0)

        if l_max is not None:  # This slice contains some lung
            binary_nda[i][labeling != l_max] = 1
        binary_nda[i] -= 1
        image_slice = sitk.GetImageFromArray(binary_nda[i])
        image_slice = sitk.BinaryFillhole(image_slice)
        binary_nda[i] = sitk.GetArrayFromImage(image_slice)

    binary_image = sitk.GetImageFromArray(binary_nda)
    binary_image.CopyInformation(image)

    return binary_image


def segment_lung(image, fill_lung_structures=True, crop_ratio=0.05):
    nda = sitk.GetArrayFromImage(image)
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    shape = np.shape(nda)
    pad_width = int(shape[1] * crop_ratio)
    nda = nda[:, pad_width:-pad_width, :]

    nda[nda == nda[0, 0, 0]] = -1000
    binary_nda = np.array(nda > -320, dtype=np.int8) + 1
    labels = measure.label(binary_nda)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    # for index in [[0, 0, 0], [shape[0] - 1, 0, 0], [shape[0] - 1, 0, 0], [10, int(shape[1] / 2), 0],
    #             [0, 0, int(shape[1] / 2)]]:
    for index in [[0, 0, 0], ]:
        i, j, k = index
        background_label = labels[i, j, k]
        # Fill the air around the person
        binary_nda[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(tqdm(binary_nda)):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_nda[i][labeling != l_max] = 1

    binary_nda -= 1  # Make the image actual binary
    binary_nda = 1 - binary_nda # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_nda, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_nda[labels != l_max] = 0

    binary_nda = np.pad(binary_nda, pad_width=pad_width, mode='constant', constant_values=0)[pad_width:-pad_width,
                   :, pad_width:-pad_width]
    # print(binary_image.shape)

    binary_image = sitk.GetImageFromArray(binary_nda)
    binary_image.CopyInformation(image)
    return binary_image


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def gauss_map_3d(shape, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z):
    # important: we assume that shape is the order in nda array returned by sitk.GetArrayFromImage
    # meaning it is (k,j,i)
    size_z, size_y, size_x = shape
    assert isinstance(size_x, int)
    assert isinstance(size_y, int)
    assert isinstance(size_z, int)

    x = np.arange(0, size_x, dtype=float)
    y = np.arange(0, size_y, dtype=float)[:, np.newaxis]
    z = np.arange(0, size_z, dtype=float)[:, np.newaxis, np.newaxis]

    exp_part = (x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2) +  \
               (z - mu_z) ** 2 / (2 * sigma_z ** 2)
    return 1 / (2 * np.pi * sigma_x * sigma_y * sigma_z) * np.exp(-exp_part)


def np_dice_coef(y_true, y_pred, threshold=0.5):
    if np.amax(y_pred) > 1 or np.amax(y_true) > 1:
        print('warning: values must be between 0 and 1!')
    smooth = 1.
    y_th = np.copy(y_pred)
    y_th[y_th >= threshold] = 1
    y_th[y_th < threshold] = 0
    y_true_f = y_true.flatten()
    y_pred_f = y_th.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def fliplr_2d_and_3d(nda):
    nda_f = np.zeros_like(nda)
    if nda.ndim == 2:
        nda_f = np.fliplr(nda)
        return nda_f
    elif nda.ndim == 3:
        for j in range(nda.shape[0]):
            nda_f[j, :, :] = np.fliplr(nda[j, :, :])
        return nda_f
    else:
        print('warning: the numpy array dim must be either 2 or 3.')


def label_one_hot(nda, n_classes):
    shape = nda.shape
    one_hot_nda = np.zeros((shape[0], shape[1], shape[2], n_classes))
    for i in range(n_classes):
        tmp = np.copy(nda)
        tmp[tmp != i + 1] = 255
        tmp[tmp == i + 1] = 1
        one_hot_nda[..., i] = tmp
    one_hot_nda[one_hot_nda == 255] = 0
    return one_hot_nda


def resample(image, output_spacing, interplator):
    resample = sitk.ResampleImageFilter()
    input_size = image.GetSize()
    pixel_type = image.GetPixelID()
    input_spacing = image.GetSpacing()
    input_spacing = np.round(input_spacing, 5)
    output_size = [int(np.floor(input_spacing[0] / output_spacing[0] * input_size[0])),
                   int(np.floor(input_spacing[1] / output_spacing[1] * input_size[1])),
                   int(np.floor(input_spacing[2] / output_spacing[2] * input_size[2]))]
    resample.SetInterpolator(interplator)
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(output_size)
    resample.SetOutputPixelType(pixel_type)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    return resample.Execute(image)


def pad_crop(image, output_size):
    size = image.GetSize()
    if size[0] < output_size[0]:
        image = pad_image(image, (output_size[0], size[1], size[2]))
    if size[1] < output_size[1]:
        image = pad_image(image, (size[0], output_size[1], size[2]))
    if size[2] < output_size[2]:
        image = pad_image(image, (size[0], size[1], output_size[2]))
    image = crop_image(image, (output_size[0], output_size[1], output_size[2]))
    return image


def crop_image(image, output_size):
    input_size = image.GetSize()
    size_diff = np.subtract(input_size, output_size)
    size_diff[size_diff < 0] = 0
    if np.sum(size_diff) == 0:
        return image
    lower_bound = [int(diff / 2) for diff in size_diff]
    upper_bound = size_diff - lower_bound
    upper_bound = [int(item) for item in upper_bound]
    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize(lower_bound)
    crop.SetUpperBoundaryCropSize(upper_bound)
    return crop.Execute(image)


def pad_image(image, output_size):
    input_size = image.GetSize()
    size_diff = np.subtract(output_size, input_size)
    size_diff[size_diff < 0] = 0
    lower_bound = [int(diff / 2) for diff in size_diff]
    upper_bound = size_diff - lower_bound
    upper_bound = [int(item) for item in upper_bound]
    padding = sitk.ConstantPadImageFilter()
    padding.SetPadLowerBound(lower_bound)
    padding.SetPadUpperBound(upper_bound)
    padding.SetConstant(float(np.amin(sitk.GetArrayFromImage(image))))
    return padding.Execute(image)


def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes



class SegmentationQualityMetrics:
    """Code taken from https://github.com/hjkuijf/MRBrainS18 with minor refactoring"""

    def __init__(self, labels, test_image, pred_image):
        self.test_image = test_image
        self.pred_image = pred_image
        self.labels = labels
        """
        labels example
        #
        labels = {1: 'Cortical gray matter',
          2: 'Basal ganglia',
          3: 'White matter',
          4: 'White matter lesions',
          5: 'Cerebrospinal fluid in the extracerebral space',
          6: 'Ventricles',
          7: 'Cerebellum',
          8: 'Brain stem',
          # The two labels below are ignored:
          #9: 'Infarction',
          #10: 'Other',
          }
        """

    def get_dice(self):
        """Compute the Dice Similarity Coefficient."""
        dsc = dict()
        for k in self.labels.keys():
            test_nda = sitk.GetArrayFromImage(sitk.BinaryThreshold(self.test_image, k, k, 1, 0)).flatten()
            pred_nda = sitk.GetArrayFromImage(sitk.BinaryThreshold(self.pred_image, k, k, 1, 0)).flatten()
            # similarity = 1.0 - dissimilarity
            # spatial.distance.dice raises a ZeroDivisionError if both arrays contain only zeros.
            try:
                dsc[k] = 1.0 - spatial.distance.dice(test_nda, pred_nda)
            except ZeroDivisionError:
                dsc[k] = None

        return dsc

    def get_hausdorff(self):
        """Compute the 95% Hausdorff distance."""
        hd = dict()
        for k in self.labels.keys():
            l_test_image = sitk.BinaryThreshold(self.test_image, k, k, 1, 0)
            l_pred_image = sitk.BinaryThreshold(self.pred_image, k, k, 1, 0)

            # Hausdorff distance is only defined when something is detected
            statistics = sitk.StatisticsImageFilter()
            statistics.Execute(l_test_image)
            l_test_sum = statistics.GetSum()
            statistics.Execute(l_pred_image)
            l_result_sum = statistics.GetSum()
            if l_test_sum == 0 or l_result_sum == 0:
                hd[k] = None
                continue

            # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions.
            # Erosion is performed in 2D
            e_test_image = sitk.BinaryErode(l_test_image, (1, 1, 0))
            e_pred_image = sitk.BinaryErode(l_pred_image, (1, 1, 0))

            h_test_image = sitk.Subtract(l_test_image, e_test_image)
            h_pred_image = sitk.Subtract(l_pred_image, e_pred_image)

            h_test_nda = sitk.GetArrayFromImage(h_test_image)
            h_pred_nda = sitk.GetArrayFromImage(h_pred_image)

            # Convert voxel location to world coordinates. Use the coordinate system of the test image
            # np.nonzero   = elements of the boundary in numpy order (zyx)
            # np.flipud    = elements in xyz order
            # np.transpose = create tuples (x,y,z)
            # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
            # (Simple)ITK does not accept all Numpy arrays; therefore we need to convert the coordinate
            # tuples into a Python list before passing them to TransformIndexToPhysicalPoint().
            testCoordinates = [self.test_image.TransformIndexToPhysicalPoint(x.tolist()) for x in
                               np.transpose(np.flipud(np.nonzero(h_test_nda)))]
            resultCoordinates = [self.test_image.TransformIndexToPhysicalPoint(x.tolist()) for x in
                                 np.transpose(np.flipud(np.nonzero(h_pred_nda)))]

            # Use a kd-tree for fast spatial search
            def get_distance_from_a_to_b(a, b):
                kdTree = spatial.KDTree(a, leafsize=100)
                return kdTree.query(b, k=1, eps=0, p=2)[0]

            # Compute distances from test to result and vice versa.
            d_test_to_pred = get_distance_from_a_to_b(testCoordinates, resultCoordinates)
            d_pred_to_test = get_distance_from_a_to_b(resultCoordinates, testCoordinates)
            hd[k] = max(np.percentile(d_test_to_pred, 95), np.percentile(d_pred_to_test, 95))

        return hd

    def get_vs(self):
        """Volume similarity.

        VS = 1 - abs(A - B) / (A + B)

        A = ground truth in ML
        B = participant segmentation in ML
        """
        # Compute statistics of both images
        test_stats = sitk.StatisticsImageFilter()
        pred_stats = sitk.StatisticsImageFilter()

        vs = dict()
        for k in self.labels.keys():
            test_stats.Execute(sitk.BinaryThreshold(self.test_image, k, k, 1, 0))
            pred_stats.Execute(sitk.BinaryThreshold(self.pred_image, k, k, 1, 0))

            num = abs(test_stats.GetSum() - pred_stats.GetSum())
            denom = test_stats.GetSum() + pred_stats.GetSum()

            if denom > 0:
                vs[k] = 1 - float(num) / denom
            else:
                vs[k] = None

        return vs
