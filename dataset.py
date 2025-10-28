import os
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

class FruitStateDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):

        self.root = root
        self.split = split
        self.transform = transform

        fruit_order = ['Banana', 'Maça', 'Tomates']   
        state_order = ['Old', 'Ripe', 'Unripe']     

        self.fruit_classes = [f for f in fruit_order if os.path.isdir(os.path.join(root, f))]
        self.fruit_to_idx = {f: i for i, f in enumerate(self.fruit_classes)}

        sample_states = set()
        
        for fruit in self.fruit_classes:
            candidate = os.path.join(root, fruit, split)
            if os.path.isdir(candidate):
                for st in os.listdir(candidate):
                    if os.path.isdir(os.path.join(candidate, st)):
                        sample_states.add(st)
        sample_states = sorted(sample_states)


        self.state_classes = [s for s in state_order if s in sample_states]
        self.state_to_idx = {s: i for i, s in enumerate(self.state_classes)}

        # Agora varre arquivos
        self.samples = [] 
        for fruit in self.fruit_classes:
            split_dir = os.path.join(root, fruit, split)
            if not os.path.isdir(split_dir):
                continue
            for state in self.state_classes:
                state_dir = os.path.join(split_dir, state)
                if not os.path.isdir(state_dir):
                    continue
                
                for ext in IMAGE_EXTS:
                    for p in glob(os.path.join(state_dir, f"*{ext}")):
                        self.samples.append((p, self.fruit_to_idx[fruit], self.state_to_idx[state]))
               

        if len(self.samples) == 0:
            raise RuntimeError(f"Nenhuma imagem encontrada em {root} para split='{split}'. Verifique a estrutura.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fruit_idx, state_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, torch.tensor(fruit_idx, dtype=torch.long), torch.tensor(state_idx, dtype=torch.long)
