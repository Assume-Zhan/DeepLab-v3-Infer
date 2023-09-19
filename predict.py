import network
import os
import time

from torch.utils.data import dataset
from tqdm import tqdm
from utils import set_bn_momentum

from datasets import VOCSegmentation, Cityscapes
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image

def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def predict(image_list):
    
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

    # Inference
    result_list = []
    time_diff = 0
    with torch.no_grad():
        model = model.eval()
        for image in tqdm(image_list):
            image_copy = image

            image = transform(image).unsqueeze(0)
            image = image.to(device)

            t1 = time_synchronized()
            pred = model(image).max(1)[1].cpu().numpy()[0]
            t2 = time_synchronized()

            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)

            blend_image = Image.blend(image_copy, colorized_preds, alpha = 0.7)

            result_list.append(blend_image)
            time_diff += (t2 - t1)
            
    image_length = len(image_list)
            
    print(f"Total inference time : {time_diff:.3f} for {image_length} images")
    if time_diff:
        print(f"Total fps : {(image_length / time_diff):.3f}")

    return result_list

if __name__ == '__main__':
    
    image_path = "./samples/1_image.png"
    image_list = []
    
    image_list.append(Image.open(image_path).convert('RGB'))
    
    result_images = predict(image_list)
    
    for image in result_images:
        image.save("./test_results/result.png")
