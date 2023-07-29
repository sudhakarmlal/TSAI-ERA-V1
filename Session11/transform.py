import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_a_train_transform():
    """Get transformer for training data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.PadIfNeeded(40,40),
        A.RandomCrop(32,32),
        A.HorizontalFlip(p=0.5),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2443, 0.2408, 0.2581)), #normalize
        # A.Cutout(8,8,8),
        A.CoarseDropout(1,16,16, p=0.5),
        ToTensorV2()])


def get_a_test_transform():
    """Get transformer for test data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.Normalize((0.4942, 0.4851, 0.4504), (0.2439, 0.2402, 0.2582)), #apply normalization
        ToTensorV2()])
