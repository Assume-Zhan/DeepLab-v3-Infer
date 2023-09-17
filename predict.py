import network
import os

from torch.utils.data import dataset
from tqdm import tqdm
from utils import set_bn_momentum

from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image

def main():
    
    USE_CITYSCAPES = 0
    input_file = "samples/23_image.png"
    model_name = "deeplabv3plus_mobilenet"
    ckpt = "checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth"
    
    save_dir = "test_results"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
    if USE_CITYSCAPES:
        decode_fn = Cityscapes.decode_target      # 19
        nc = 19
    else:
        decode_fn = VOCSegmentation.decode_target # 21
        nc = 21

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    image_files.append(input_file)
    
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
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0)
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0]
            
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if save_dir:
                colorized_preds.save(os.path.join(save_dir, img_name + '.png'))

if __name__ == '__main__':
    main()
