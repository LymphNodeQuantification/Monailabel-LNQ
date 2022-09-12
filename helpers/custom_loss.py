from keras import backend as K

'''
def dice(y_true, y_predicted):
    smooth = 1e-4
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_predicted)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def supervised_dice_metric(y_true, y_pred):
    y_true_supervised = y_true[..., 0] * y_true[..., 1]
    y_pred_supervised = y_pred[..., 0] * y_true[..., 1]
    return dice(y_true_supervised, y_pred_supervised)
'''


class PartialCE:
    def __init__(self, n_classes, class_weights):
        self.class_weights = class_weights
        self.n_classes = n_classes

    def loss(self, y_true, y_pred):
        weights = y_true[..., 0] * self.class_weights[0]
        for i in range(1, self.n_classes):
            weights += y_true[..., i] * self.class_weights[i]
        weights_list = list()
        for i in range(self.n_classes):
            weights_list.append(weights)
        weights = K.stack(weights_list, axis=-1)
        loss = - y_true * K.log(y_pred) * weights
        # return K.sum(loss) / K.sum(y_true)
        return K.sum(loss)


class UnsupervisedDiceLoss:
    def __init__(self, n_classes, smooth):
        self.n_classes = n_classes
        self.smooth = smooth

    def dice(self, y_true, y_predicted):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_predicted)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return dice

    def loss(self, y_true, y_pred):
        # y_true components: y_supervised, y_hat, supervised_mask, unsupervised_mask, y_hat_weight:
        loss = 0
        for i in range(self.n_classes):
            y_true_supervised = y_true[..., i] * y_true[..., -3]
            y_pred_supervised = y_pred[..., i] * y_true[..., -3]
            supervised_dice = self.dice(y_true_supervised, y_pred_supervised)
            #
            y_true_hat = y_true[..., self.n_classes + i] * (y_true[..., -2])
            y_pred_hat = y_pred[..., self.n_classes + i] * (y_true[..., -2])
            us_dice = self.dice(y_true_hat, y_pred_hat)
            #
            y_hat_weight = K.flatten(y_true[..., i, -1])[0]
            loss -= supervised_dice + us_dice * y_hat_weight
            loss -= self.dice(y_true_supervised, y_pred_supervised)
        return loss / self.n_classes


class UnsupervisedLabelDice:
    def __init__(self, label_value, smooth):
        self.label_value = label_value
        self.smooth = smooth

    def dice(self, y_true, y_pred):
        y_true_f = K.flatten(y_true[..., self.label_value] * y_true[..., -3])
        y_pred_f = K.flatten(y_pred[..., self.label_value] * y_true[..., -3])
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        return (2. * intersection + self.smooth) / (union + self.smooth)


class LabelDice:
    def __init__(self, label_value, smooth):
        self.label_value = label_value
        self.smooth = smooth

    def dice(self, y_true, y_pred):
        y_true_f = K.flatten(y_true[..., self.label_value])
        y_pred_f = K.flatten(y_pred[..., self.label_value])
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        return (2. * intersection + self.smooth) / (union + self.smooth)


class DiceLoss:
    def __init__(self, n_classes, smooth):
        self.n_classes = n_classes
        self.smooth = smooth

    def dice(self, y_true, y_predicted):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_predicted)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return dice

    def loss(self, y_true, y_pred):
        loss = 0
        for i in range(self.n_classes):
            loss -= self.dice(y_true[..., i], y_pred[..., i])
        return loss


class PartialDiceLoss:
    def __init__(self, n_classes, class_weights, smooth):
        self.class_weights = class_weights
        self.n_classes = n_classes
        self.smooth = smooth

    def dice(self, y_true, y_predicted):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_predicted)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return dice

    def loss(self, y_true, y_pred):
        loss = 0
        mask = K.sum(y_true, axis=-1)
        for i in range(self.n_classes):
            y_true_supervised = y_true[..., i] * mask
            y_pred_supervised = y_pred[..., i] * mask
            partial_dice = self.dice(y_true_supervised, y_pred_supervised)
            loss -= self.class_weights[i] * partial_dice
        return loss / self.n_classes


class PartialLabelDice:
    def __init__(self, label_value):
        self.label_value = label_value

    def dice(self, y_true, y_pred):
        epsilon = 1e-3
        mask = K.sum(y_true, axis=-1)
        y_true_f = K.flatten(y_true[..., self.label_value] * mask)
        y_pred_f = K.flatten(y_pred[..., self.label_value] * mask)
        y_pred_f = K.cast(K.greater(K.clip(y_pred_f, 0, 1), 0.5), K.floatx())
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        return (2. * intersection + epsilon) / (union + epsilon)


def dice_3forground(y_true, y_pred):
    smooth = .1
    dice = 0
    for i in range(1, 4):
        y_true_f = K.flatten(y_true[..., i])
        threshold_value = 0.5
        y_pred_f = K.flatten(y_pred[..., i])
        y_pred_f = K.cast(K.greater(K.clip(y_pred_f, 0, 1), threshold_value), K.floatx())
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        dice += (2. * intersection + smooth) / (union + smooth)
    return dice / 3