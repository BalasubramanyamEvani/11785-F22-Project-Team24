import json, os
import torch
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def open_data_file(path):
    with open(path, "r") as fd:
        data = [line.split("\t") for line in fd.readlines()]
        compact = []
        i = 0
        temp = []
        while i < len(data):
            temp = []
            for j in range(5):
                temp.append(data[i+j][1])
            compact.append((data[i][0], temp))
            i += 5
        return compact


class FlickerCaptionDataset(Dataset):
    def __init__(self, transforms, data_path, vocab, adversarial=False, dataset="flicker8k", img_src="Flicker8k_Dataset", partition="train"):
        super(FlickerCaptionDataset, self).__init__()
        self.partition = partition
        self.transforms = transforms
        self.image_src_path = os.path.join(os.getcwd(), dataset, img_src)
        self.data = None
        self.adversarial = adversarial
        self.vocab = vocab
        self.data = open_data_file(os.path.join(os.getcwd(), dataset, "prep-test.txt"))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, captions = self.data[index]

        if not self.adversarial:
            img = pil_loader(os.path.join(self.image_src_path, image))
        else:
            img = pil_loader(os.path.join(self.image_src_path, "adversarial", image))
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        res_captions = []
        for caption in captions:
            caption_indx = []
            for token in caption.split():
                caption_indx.append(self.vocab[token])
            res_captions.append(caption_indx)

        return torch.FloatTensor(img), res_captions
