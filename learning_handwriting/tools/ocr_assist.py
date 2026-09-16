import argparse
from paddleocr import PaddleOCR
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);a=p.parse_args();ocr=PaddleOCR(lang='korean');r=ocr.predict(a.image)
 for item in r:
  print(item)
if __name__=='__main__':main()
