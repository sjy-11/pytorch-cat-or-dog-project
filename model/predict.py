from model.model import Net
import torch
import torchvision.transforms as transforms
from PIL import Image

def get_image_prediction(image_path):
    model = Net()
    model.load_state_dict(torch.load("./model/model_state_dict.pth"))
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    
    classes = ['Cat', 'Dog']

    if confidence.item() >= 0.5:
        return classes[pred.item()], confidence.item()
    else:
        return "Not a Cat or Dog", 1 - confidence.item()