import json
import requests
import io
from PIL import Image
import time
from pikepdf import Pdf, PdfImage, Name
from stages.pdfs_to_images import pdfs_to_images

start = time.time()
session = requests.Session()

for _ in range(10):
    data = session.get('https://chroniclingamerica.loc.gov/data/batches/ak_albatross_ver01/data/sn84020657/0027952665A/1916030101/0007.jp2')
    data = data.content
    im = Image.open(io.BytesIO(data))

print(f'{((time.time() - start)/10):3f} seconds elapsed')
print(im.size)

start = time.time()
for _ in range(10):
    data = session.get('https://chroniclingamerica.loc.gov/data/batches/ak_albatross_ver01/data/sn84020657/0027952665A/1916030101/0007.pdf')
    pdf_img = Pdf.open(io.BytesIO(data.content))
    page1 = pdf_img.pages[0]
    relevant_key = [key for key in page1.images.keys()][0]
    rawimage = page1.images[relevant_key]
    pdfimage = PdfImage(rawimage)
    image = pdfimage.as_pil_image()

print(f'{((time.time() - start)/10):3f} seconds elapsed')
print(image.size)