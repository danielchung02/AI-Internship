from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from .encoding import decompose

class GlyphDataset(Dataset):
    def __init__(self, csv_path, char_to_id, size=96, augment=False):
        self.rows = pd.read_csv(csv_path)
        self.root = Path(csv_path).parent
        self.char_to_id, self.size, self.augment = char_to_id, size, augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows.iloc[index]; ch = str(row.char)
        img = Image.open(self.root / row.path).convert('L')
        img = ImageOps.fit(img, (self.size, self.size), method=Image.Resampling.LANCZOS, centering=(.5,.5))
        if self.augment:
            from torchvision.transforms import functional as F
            img = F.affine(img, angle=float(torch.empty(1).uniform_(-2,2)), translate=[int(torch.randint(-2,3,(1,))),int(torch.randint(-2,3,(1,)))], scale=1, shear=0)
        x = 1 - torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(1,self.size,self.size)/255
        # Ink mask: paper white -> 0, ink -> 1
        parts = decompose(ch)
        if parts: ini,vow,fin=parts; other=0; hangul=1
        else: ini=vow=fin=0; other=self.char_to_id[ch]; hangul=0
        return x, torch.tensor([ini,vow,fin,other,hangul], dtype=torch.long)
