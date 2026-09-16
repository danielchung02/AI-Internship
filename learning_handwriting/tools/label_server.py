"""Local web labeler: OCR drafts -> character crops + data/labels.csv.

Run on the GPU server bound to 127.0.0.1 and access through an SSH tunnel.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
RAW, OCR, CROPS, CSV = ROOT/'data/raw', ROOT/'data/ocr_easyocr', ROOT/'data/crops', ROOT/'data/labels.csv'
app = FastAPI(); CROPS.mkdir(parents=True, exist_ok=True)

class Crop(BaseModel):
    page: str; x: float; y: float; w: float; h: float; char: str
class Split(BaseModel):
    page: str; x: float; y: float; w: float; h: float; text: str

def safe_page(name: str) -> Path:
    path = (RAW / name).resolve()
    if path.parent != RAW.resolve() or not path.is_file(): raise HTTPException(404, 'Image not found')
    return path
def save_crop(page, x, y, w, h, char):
    char = char.strip()
    if len(char) != 1: raise HTTPException(400, 'Label must be exactly one character')
    src = safe_page(page); im = Image.open(src).convert('RGB'); x,y,w,h = map(lambda n:max(0,int(round(n))), (x,y,w,h))
    if w < 4 or h < 4: raise HTTPException(400, 'Crop is too small')
    x2,y2=min(im.width,x+w),min(im.height,y+h)
    if x >= x2 or y >= y2: raise HTTPException(400, 'Crop is outside image')
    base=re.sub(r'[^A-Za-z0-9_]+','_',Path(page).stem); index=len(list(CROPS.glob(base+'_*'))) + 1; name=f'{base}_{index:06d}.png'
    im.crop((x,y,x2,y2)).save(CROPS/name)
    new=not CSV.exists()
    with CSV.open('a',newline='',encoding='utf-8') as f:
        wr=csv.writer(f)
        if new: wr.writerow(['path','char'])
        wr.writerow([f'crops/{name}',char])
    return name

@app.get('/', response_class=HTMLResponse)
def home(): return HTML
@app.get('/api/pages')
def pages(): return [p.name for p in sorted(RAW.iterdir()) if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
@app.get('/api/ocr/{page}')
def ocr(page: str):
    path=OCR/(Path(page).stem+'.json')
    if not path.exists(): return {'results': []}
    return json.loads(path.read_text(encoding='utf-8'))
@app.get('/image/{page}')
def image(page: str): return FileResponse(safe_page(page))
@app.post('/api/crop')
def crop(item: Crop): return {'file':save_crop(**item.model_dump())}
@app.post('/api/split')
def split(item: Split):
    chars=[c for c in item.text.strip() if not c.isspace()]
    if not chars: raise HTTPException(400,'No characters to split')
    saved=[]
    for i,ch in enumerate(chars): saved.append(save_crop(item.page,item.x+item.w*i/len(chars),item.y,item.w/len(chars),item.h,ch))
    return {'saved':saved}

HTML = r'''<!doctype html><meta charset="utf-8"><title>손글씨 라벨러</title><style>body{font:14px system-ui,'Malgun Gothic';margin:0;background:#f2f4f6;color:#17212b}header{padding:12px 18px;background:#17365d;color:white;position:sticky;top:0}main{display:grid;grid-template-columns:minmax(500px,1fr) 330px;gap:14px;padding:14px}.card{background:white;padding:12px;border-radius:8px}canvas{max-width:100%;border:1px solid #aaa;cursor:crosshair}button,input{font:inherit;padding:7px;margin:3px}button{background:#176c58;color:#fff;border:0;border-radius:5px}.item{border-top:1px solid #ddd;padding:7px;cursor:pointer}.item:hover{background:#eef7f4}.small{color:#52606d;font-size:12px}</style><header><b>손글씨 라벨러</b> · OCR 결과는 초안입니다. 자동 분할 후 반드시 표본을 눈으로 검수하세요.</header><main><div class="card"><button onclick="prev()">← 이전</button><button onclick="next()">다음 →</button><span id="name"></span><p class="small">직접 라벨: 이미지에서 글자 하나를 드래그 → 글자 입력 → 저장. 오른쪽 OCR 항목을 누르면 그 영역과 텍스트가 선택됩니다.</p><canvas id="cv"></canvas></div><div class="card"><b>직접 글자 크롭</b><br><input id="char" maxlength="1" placeholder="글자 1개"><button onclick="saveCrop()">선택 영역 저장</button><hr><b>OCR 박스 자동 분할</b><br><input id="word" placeholder="OCR 텍스트 수정"><button onclick="splitWord()">글자별 자동 분할</button><p class="small">공백은 제외합니다. 영어 필기 연결·수식·붙어 쓴 글자는 자동 분할이 틀릴 수 있으니 직접 크롭을 쓰세요.</p><hr><b>OCR 초안</b><div id="items"></div></div></main><script>let pages=[],n=0,img=new Image(),sel=null,drag=null,scale=1;const cv=document.querySelector('#cv'),ctx=cv.getContext('2d'),$=x=>document.querySelector(x);async function init(){pages=await (await fetch('/api/pages')).json();load()}async function load(){if(!pages.length)return;let p=pages[n];$('#name').textContent=`${n+1}/${pages.length} ${p}`;img.onload=()=>{scale=Math.min(1,900/img.width);cv.width=img.width*scale;cv.height=img.height*scale;draw()};img.src='/image/'+encodeURIComponent(p);let d=await (await fetch('/api/ocr/'+encodeURIComponent(p))).json();let box=$('#items');box.innerHTML='';(d.results||[]).forEach(r=>{let e=document.createElement('div');e.className='item';e.textContent=`${r.text}  (${Math.round(r.confidence*100)}%)`;e.onclick=()=>pick(r);box.append(e)})}function draw(){ctx.drawImage(img,0,0,cv.width,cv.height);if(sel){ctx.strokeStyle='#e21b32';ctx.lineWidth=2;ctx.strokeRect(sel.x*scale,sel.y*scale,sel.w*scale,sel.h*scale)}}function pick(r){let xs=r.polygon.map(p=>p[0]),ys=r.polygon.map(p=>p[1]);sel={x:Math.min(...xs),y:Math.min(...ys),w:Math.max(...xs)-Math.min(...xs),h:Math.max(...ys)-Math.min(...ys)};$('#word').value=r.text;draw()}cv.onmousedown=e=>{let r=cv.getBoundingClientRect();drag={x:(e.clientX-r.left)/scale,y:(e.clientY-r.top)/scale}};cv.onmouseup=e=>{if(!drag)return;let r=cv.getBoundingClientRect(),x=(e.clientX-r.left)/scale,y=(e.clientY-r.top)/scale;sel={x:Math.min(drag.x,x),y:Math.min(drag.y,y),w:Math.abs(x-drag.x),h:Math.abs(y-drag.y)};drag=null;draw()};async function post(url,body){let r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let d=await r.json();if(!r.ok)alert(d.detail||'저장 실패');else alert('저장됨: '+(d.file||d.saved.length+'개'))}function saveCrop(){if(!sel)return alert('영역을 먼저 선택하세요.');post('/api/crop',{page:pages[n],...sel,char:$('#char').value})}function splitWord(){if(!sel)return alert('OCR 항목을 선택하세요.');post('/api/split',{page:pages[n],...sel,text:$('#word').value})}function prev(){n=Math.max(0,n-1);load()}function next(){n=Math.min(pages.length-1,n+1);load()}init()</script>'''
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8001);a=p.parse_args();uvicorn.run(app,host='127.0.0.1',port=a.port)
