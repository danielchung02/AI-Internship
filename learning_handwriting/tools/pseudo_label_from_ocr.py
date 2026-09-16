"""Convert EasyOCR word boxes into deliberately noisy character-level pseudo labels.

For rapid pipeline experiments only: each OCR box is divided evenly by its
recognized Unicode characters. Outputs are isolated from manually verified data.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--ocr',type=Path,default=ROOT/'data/ocr_easyocr');p.add_argument('--out',type=Path,default=ROOT/'data/crops_pseudo');p.add_argument('--csv',type=Path,default=ROOT/'data/labels_pseudo.csv');p.add_argument('--min-confidence',type=float,default=.2);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True);a.csv.parent.mkdir(parents=True,exist_ok=True); saved=skipped=0
 with a.csv.open('w',newline='',encoding='utf8') as f:
  wr=csv.writer(f);wr.writerow(['path','char'])
  for jf in sorted(a.ocr.glob('*.json')):
   data=json.loads(jf.read_text(encoding='utf8'));src=Path(data['source'])
   if not src.exists(): print('missing:',src);continue
   im=Image.open(src).convert('RGB');stem=re.sub(r'[^A-Za-z0-9_]+','_',src.stem)
   for row in data.get('results',[]):
    chars=[c for c in str(row.get('text','')) if not c.isspace()]
    if float(row.get('confidence',0))<a.min_confidence or not chars: skipped+=1;continue
    xs=[q[0] for q in row['polygon']];ys=[q[1] for q in row['polygon']];x0,x1=max(0,int(min(xs))),min(im.width,int(max(xs)));y0,y1=max(0,int(min(ys))),min(im.height,int(max(ys)))
    if x1-x0<4 or y1-y0<4: skipped+=1;continue
    for i,ch in enumerate(chars):
     left=x0+(x1-x0)*i//len(chars);right=x0+(x1-x0)*(i+1)//len(chars)
     if right-left<3: continue
     name=f'{stem}_{saved:07d}.png';im.crop((left,y0,right,y1)).save(a.out/name);wr.writerow([f'{a.out.name}/{name}',ch]);saved+=1
 print(f'Pseudo labels saved: {saved}; skipped OCR boxes: {skipped}');print('CSV:',a.csv)
if __name__=='__main__':main()
