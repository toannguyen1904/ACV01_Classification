import torch
from collections import OrderedDict
import imageio
from models.resnet import resnet50
from PIL import Image
from torchvision import transforms
from config import input_size
import numpy as np


device = torch.device("cuda")

checkpoint_path = "./log/resnet50/best_model.pth"   # checkpoint of the best model
image_path = 'PATH_TO_YOUR_TEST_IMAGE'

""" Load an image. You can refer to dataset/pre_data.py to test for multiple images (using dataloader)"""
img = imageio.imread(image_path)
if len(img.shape) == 2:
    img = np.stack([img] * 3, 2)
img = Image.fromarray(img, mode='RGB')
img = transforms.Resize(input_size + 16)(img)
img = transforms.CenterCrop(input_size)(img)
img = transforms.ToTensor()(img)
img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
img = img.to(device).unsqueeze(0)

model = resnet50(pth_url='', pretrained=False)

""" Load the checkpoint """
checkpoint = torch.load(checkpoint_path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

print("Model loaded!")
model = model.to(device)

_, logits = model(img)
preds = logits.max(1, keepdim=True)[1]

# Print prediction
print('Prediction: ' + str(preds.item()))