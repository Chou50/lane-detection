from torch.utils.data import Dataset
import cv2


class SimDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        super().__init__()
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx])

        # resize
        image = cv2.resize(image, (512, 256))
        mask = cv2.resize(mask, (512, 256))

        if self.transform:
            image = self.transform(image)
        return image, mask


