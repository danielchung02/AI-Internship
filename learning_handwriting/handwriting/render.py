from io import BytesIO
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas as pdf_canvas
from .encoding import decompose
from .model import JamoGlyphNet

class Handwriter:
 def __init__(self, checkpoint, device='cpu'):
  data=torch.load(checkpoint,map_location=device,weights_only=False);self.device=torch.device(device);self.ids=data['char_to_id'];self.size=data.get('image_size',96);self.model=JamoGlyphNet(data['other_vocab']).to(self.device);self.model.load_state_dict(data['state_dict']);self.model.eval()
 def glyph(self,ch,height=64):
  p=decompose(ch)
  if p: values=(*p,0,1)
  elif ch in self.ids: values=(0,0,0,self.ids[ch],0)
  else: return None
  c=torch.tensor(values,device=self.device).view(1,5)
  with torch.no_grad(): y=self.model(*[c[:,i] for i in range(5)])[0,0].cpu()
  alpha=Image.fromarray((y.numpy()*255).astype('uint8')).resize((height,height),Image.Resampling.LANCZOS)
  ink=Image.new('RGBA',(height,height),(20,40,105,0));ink.putalpha(alpha);return ink
 def page(self,text,width=2480,height=3508,margin=150,font_height=82,line_height=120):
  paper=Image.new('RGBA',(width,height),(255,253,247,255));x=y=margin
  for ch in text:
   if ch=='\n':x=margin;y+=line_height;continue
   if ch==' ':x+=font_height//2;continue
   if x+font_height>width-margin:x=margin;y+=line_height
   if y+font_height>height-margin:break
   g=self.glyph(ch,font_height)
   if g: paper.alpha_composite(g,(x,y))
   else: ImageDraw.Draw(paper).rectangle((x,y,x+font_height*.75,y+font_height*.75),outline=(180,40,40,255),width=2)
   x+=font_height+8
  return paper
 def pdf(self,text):
  page=self.page(text); b=BytesIO();c=pdf_canvas.Canvas(b,pagesize=(page.width,page.height));im=BytesIO();page.convert('RGB').save(im,'PNG');c.drawImage(__import__('reportlab').lib.utils.ImageReader(im),0,0,page.width,page.height);c.showPage();c.save();return b.getvalue()
