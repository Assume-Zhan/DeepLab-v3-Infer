import cv2
import numpy as np
from convert_video import convert
from predict import predict as pt
from tqdm import tqdm

def predict_video(file_name):
    
    images = convert(file_name)
    
    # Video Writer for writing result into mp4
    video = cv2.VideoWriter("./test_results/demo1.mp4", 
                            cv2.VideoWriter_fourcc(*"mp4v"), 
                            30, (2048, 1024))

    # Pass image list to deeplab model
    final_result = pt(images)

    for image in final_result:
        video.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))

    video.release()

if __name__ == "__main__":
    predict_video("./samples/demo2.mp4")