import torchvision.transforms as transforms
def get_train_transform():
    transform = transforms.Compose(
        [transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010)))])
    return transform

def get_test_transform():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010)))])
    return transform