import json, os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, img_src_path, split_type="train"):
        super(ImageCaptionDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform
        self.img_paths = json.load(open(os.path.join(data_path, f"prep-{split_type}-img.json"), "r"))
        self.captions = json.load(open(os.path.join(data_path, f"prep-{split_type}-cap.json"), "r"))
        self.img_src_path = img_src_path

    def __getitem__(self, index):
        img_path = self.img_paths[str(index)]
        img = pil_loader(os.path.join(self.img_src_path, img_path))
        if self.transform is not None:
            img = self.transform(img)

        if self.split_type == "train":
            return torch.FloatTensor(img), torch.tensor(self.captions[str(index)])

        matching_idxs = [idx for idx, path in self.img_paths.items() if path == img_path]
        all_captions = [self.captions[idx] for idx in matching_idxs]
        return torch.FloatTensor(img), torch.tensor(self.captions[str(index)]), torch.tensor(all_captions)

    def __len__(self):
        return len(self.captions)
