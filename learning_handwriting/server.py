import argparse
from io import BytesIO
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, Response
from handwriting.render import Handwriter

HTML='''<!doctype html><meta charset="utf-8"><title>내 손글씨 출력</title><style>body{font-family:system-ui,'Malgun Gothic';max-width:760px;margin:40px auto;background:#f4f6f8}main{background:white;padding:28px;border-radius:12px}textarea{width:100%;height:280px;box-sizing:border-box;font:16px/1.6 inherit}button{padding:10px 15px;margin:12px 6px 0 0;background:#163e70;color:white;border:0;border-radius:7px}p{color:#465666}</style><main><h1>내 손글씨 출력</h1><p>ChatGPT에서 답변을 복사한 뒤 아래 버튼을 눌러 붙여넣고, 필요한 부분을 직접 검토·수정하세요.</p><button onclick="navigator.clipboard.readText().then(t=>document.querySelector('textarea').value=t).catch(()=>alert('클립보드 권한을 허용하세요.'))">클립보드 붙여넣기</button><form method="post"><textarea name="text" placeholder="여기에 원고를 붙여넣으세요"></textarea><br><button formaction="/png">PNG 다운로드</button><button formaction="/pdf">PDF 다운로드</button></form></main>'''
app=FastAPI(); writer=None
@app.get('/',response_class=HTMLResponse)
def home(): return HTML
@app.post('/png')
def png(text:str=Form(...)):
 b=BytesIO();writer.page(text).convert('RGB').save(b,'PNG');return Response(b.getvalue(),media_type='image/png',headers={'Content-Disposition':'attachment; filename=handwriting.png'})
@app.post('/pdf')
def pdf(text:str=Form(...)): return Response(writer.pdf(text),media_type='application/pdf',headers={'Content-Disposition':'attachment; filename=handwriting.pdf'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--port',type=int,default=8000);a=p.parse_args();writer=Handwriter(a.checkpoint);import uvicorn;uvicorn.run(app,host='127.0.0.1',port=a.port)
