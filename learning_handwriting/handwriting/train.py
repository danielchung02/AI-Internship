import argparse, json, random
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from .data import GlyphDataset
from .model import JamoGlyphNet

def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=120);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--device',default='cuda');a=p.parse_args()
 torch.manual_seed(42); rows=pd.read_csv(a.csv); chars=sorted(set(rows.char.astype(str))); other={c:i+1 for i,c in enumerate(c for c in chars if not ('가'<=c<='힣'))}
 ds=GlyphDataset(a.csv,other,augment=True); n=max(1,int(len(ds)*.95)); tr,va=random_split(ds,[n,len(ds)-n],generator=torch.Generator().manual_seed(42))
 dl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True); vl=DataLoader(va,batch_size=a.batch_size) if len(va) else []
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); model=JamoGlyphNet(len(other)+1).to(dev)
 # CUDA_VISIBLE_DEVICES=0,2 makes those physical GPUs appear here as cuda:0,cuda:1.
 # DataParallel is intentionally kept simple for this single-node learning project.
 if dev.type == 'cuda' and torch.cuda.device_count() > 1:
  print(f'Using {torch.cuda.device_count()} visible GPUs via DataParallel')
  model=torch.nn.DataParallel(model)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); bce=nn.BCELoss(); out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True); sample_dir=out.parent/'samples';sample_dir.mkdir(exist_ok=True)
 for epoch in range(1,a.epochs+1):
  model.train(); total=0
  for x,c in dl:
   x,c=x.to(dev),c.to(dev); y=model(c[:,0],c[:,1],c[:,2],c[:,3],c[:,4]); loss=bce(y,x)+0.35*(y-x).abs().mean();opt.zero_grad();loss.backward();opt.step();total+=loss.item()*len(x)
  print(f'epoch {epoch:03d} train={total/max(1,len(tr)):.4f}')
  if epoch%10==0:
   model.eval();x,c=next(iter(dl));
   with torch.no_grad(): pred=model(*[c[:,i].to(dev) for i in range(5)]).cpu()
   save_image(torch.cat([x[:16],pred[:16]],0),sample_dir/f'epoch_{epoch:03d}.png',nrow=16)
 weights=(model.module if isinstance(model,torch.nn.DataParallel) else model).state_dict()
 torch.save({'state_dict':weights,'other_vocab':len(other)+1,'char_to_id':other,'image_size':96},out);print('saved',out)
if __name__=='__main__':main()
