from torch.utils.data import Dataset, DataLoader, RandomSampler
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
from utils import channel_last
import torchvision.transforms as T
class JPGDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.images = list(self.root_dir.glob('*.jpg'))
        if len(self.images) == 0:
            raise Exception(f"No JPG files found in {root_dir}")

        if augment:
            self.augment = T.Compose([
                T.Resize((286,286)),
                T.RandomCrop((256, 256)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])
        else:
            self.augment = None

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        f = self.images[idx]
        img = pil_to_tensor(Image.open(f)) / 255
        if self.augment:
            img = self.augment(img)
        return img * 2 - 1

if __name__ == '__main__':
    ds = JPGDataset('./data/monet_jpg')
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    for x in dl:
        plt.figure()
        plt.imshow(channel_last((x[0:1,:,:,:]+1)/2)[0])
        plt.show()
        break
