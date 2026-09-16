"""Split a known rows×cols handwriting grid and assign labels in chars.txt order."""
import argparse,csv
from pathlib import Path
import cv2
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--chars',required=True);p.add_argument('--rows',type=int,required=True);p.add_argument('--cols',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--csv',required=True);p.add_argument('--left',type=int,default=0);p.add_argument('--top',type=int,default=0);p.add_argument('--cell-w',type=int);p.add_argument('--cell-h',type=int);a=p.parse_args()
 im=cv2.imread(a.image); chars=[x.rstrip('\n') for x in open(a.chars,encoding='utf8') if x.rstrip('\n')]; w=a.cell_w or (im.shape[1]-a.left)//a.cols;h=a.cell_h or (im.shape[0]-a.top)//a.rows;out=Path(a.out);out.mkdir(parents=True,exist_ok=True); exists=Path(a.csv).exists(); csv_root=Path(a.csv).parent.resolve(); relative_out=out.resolve().relative_to(csv_root)
 with open(a.csv,'a',newline='',encoding='utf8') as f:
  wr=csv.writer(f);
  if not exists:wr.writerow(['path','char'])
  for n,ch in enumerate(chars[:a.rows*a.cols]):
   r,c=divmod(n,a.cols);crop=im[a.top+r*h:a.top+(r+1)*h,a.left+c*w:a.left+(c+1)*w];name=f'{Path(a.image).stem}_{n:04d}.png';cv2.imwrite(str(out/name),crop);wr.writerow([str(relative_out/name),ch])
if __name__=='__main__':main()
