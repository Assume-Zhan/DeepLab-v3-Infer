import network
import os
import time
import cv2
import numpy as np

from torch.utils.data import dataset
from tqdm import tqdm
from utils import set_bn_momentum

from datasets import VOCSegmentation, Cityscapes
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image
from predict import time_synchronized

def predict_video(file_name):
    
    # Get Video Stream
    cap = cv2.VideoCapture(file_name)
    
    USE_CITYSCAPES = 1
    model_name = "deeplabv3plus_resnet101"
    ckpt = "checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"
        
    if USE_CITYSCAPES:
        decode_fn = Cityscapes.decode_target      # 19
        nc = 19
    else:
        decode_fn = VOCSegmentation.decode_target # 21
        nc = 21

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[model_name](num_classes = nc, output_stride = [8, 16])
    
    torch.backends.cudnn.benchmark = True
    
    set_bn_momentum(model.backbone, momentum = 0.01)
    
    # Load Checkpoint
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    print("Resume model from %s" % ckpt)
    del checkpoint

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])
    
    # Video Writer for writing result into mp4
    video = cv2.VideoWriter("./test_results/demo2.mp4", 
                            cv2.VideoWriter_fourcc(*"mp4v"), 
                            30, (2048, 1024))

    # Inference
    time_diff = 0
    length = 0
    with torch.no_grad():
        model = model.eval()
        while True:
            ret, frame = cap.read()
            if ret:
                length = length + 1
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image_copy = image

                image = transform(image).unsqueeze(0)
                image = image.to(device)

                t1 = time_synchronized()
                pred = model(image).max(1)[1].cpu().numpy()[0]
                t2 = time_synchronized()

                colorized_preds = decode_fn(pred).astype('uint8')
                colorized_preds = Image.fromarray(colorized_preds)

                blend_image = Image.blend(image_copy, colorized_preds, alpha = 0.7)
                
                video.write(cv2.cvtColor(np.asarray(blend_image), cv2.COLOR_RGB2BGR))
                time_diff += (t2 - t1)
                
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            
    print(f"Total inference time : {time_diff:.3f} for {length} images")
    if time_diff:
        print(f"Total fps : {(length / time_diff):.3f}")
    
    cap.release()

    video.release()

if __name__ == "__main__":
    predict_video("./samples/demo2.mp4")