import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
def main():
 p=argparse.ArgumentParser();p.add_argument('--chars',required=True);p.add_argument('--out',required=True);p.add_argument('--cols',type=int,default=10);a=p.parse_args();chars=[x.rstrip('\n') for x in open(a.chars,encoding='utf8') if x.rstrip('\n')];c=canvas.Canvas(a.out,pagesize=A4);W,H=A4;cell=W/a.cols;rows=int(H/cell)
 for i,ch in enumerate(chars):
  if i and i%(a.cols*rows)==0:c.showPage()
  n=i%(a.cols*rows);r,col=divmod(n,a.cols);x=col*cell;y=H-(r+1)*cell;c.rect(x,y,cell,cell);c.setFont('Helvetica',8);c.drawString(x+3,y+cell-12,ch)
 c.save()
if __name__=='__main__':main()
