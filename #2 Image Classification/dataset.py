import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image


class Imagenette(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        image_dir = Path(root_dir, mode)
        classes = list(image_dir.glob('*'))
        fname_list = [list(c.glob('*')) for c in classes]

        class_name_dict = {
            'n03888257': 'Parachute',
            'n02102040': 'Dog',
            'n02979186': 'Radio',
            'n03028079': 'Church',
            'n03000684': 'Chainsaw',
            'n03417042': 'Garbage Truck',
            'n01440764': 'Fish',
            'n03445777': 'Golf Ball',
            'n03425413': 'Gas Station',
            'n03394916': 'Trumpet'
        }
        self.class_list = [class_name_dict[str(c.name)] for c in classes]

        self.fnames = []
        [self.fnames.extend(l) for l in fname_list]

        self.labels = np.concatenate([np.repeat(i, len(fname_list[i])) for i in range(len(classes))])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.fnames[idx])
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, 0, 'constant')


class AddChannels:
    def __call__(self, image):
        num_channels = image.shape[0]
        if num_channels == 1:
            image = torch.cat([image, image, image], dim=0)
        return image


def get_base_transform(size):
    return transforms.Compose([
        SquarePad(),
        transforms.Resize(size),
        transforms.ToTensor(),
        AddChannels(),
        transforms.Normalize([0.3430, 0.3398, 0.3189], [0.1222, 0.1237, 0.1407])
    ])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    d = Imagenette('imagenette2', 'train', transform=get_base_transform(300))
    image, label = d[19]
    plt.imshow(image.T)
    plt.title(d.class_list[label])
    plt.show()

    l = []
    for image, label in tqdm(d):
        l.append(image.sum(-1).sum(-1))

    l = torch.stack(l)
    ppm = l.mean(0) / 300**2
    ppstd = l.std(0) / 300**2

    print(ppm, ppstd)   # -> tensor([0.3430, 0.3398, 0.3189]) tensor([0.1222, 0.1237, 0.1407])

