import cv2
import numpy
from PIL import ImageDraw, ImageFont, Image

def showText(img_OpenCV, position, s, size = 40, color = (255, 0, 0)):
    img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Black.otf', size)
    # 需要先把输出的中文字符转换成Unicode编码形式
    s = str(s.encode('utf8'), encoding = "utf-8")
    draw = ImageDraw.Draw(img_PIL)
    draw.text(xy=position, text=s, fill=color, font=font)
    img_OpenCV = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)

    return img_OpenCV