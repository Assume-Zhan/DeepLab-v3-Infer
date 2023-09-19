import cv2
import numpy as np
from PIL import Image

def convert(file_name):
    
    # Get Video Stream
    cap = cv2.VideoCapture(file_name)
    
    # Returned image list
    image_list = []
    
    while True:
        ret, cv2_im = cap.read()
        
        if ret :
            # Convert image from BGR(opencv) to RGB 
            converted = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

            pil_im = Image.fromarray(converted)
            image_list.append(pil_im)
        else:
            break
    
    cap.release()
    
    return image_list

if __name__ == "__main__":
    convert("./samples/demo1.mp4")