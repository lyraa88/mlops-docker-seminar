import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import json

# 모델 클래스 정의 (초기 모델과 동일)
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def predict_image(image_path):
    # 1. 모델 로드 (재학습 모델 우선 사용)
    model = ImprovedCNN()
    model_path = 'best_mnist_model_retrained.pth'
    if not os.path.exists(model_path):
        model_path = 'best_mnist_model.pth'
    
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        return {'error': '모델 파일이 없습니다. 초기 학습을 먼저 실행하세요.'}
    
    model.eval()

    # 2. 이미지 전처리
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 배치 차원 추가

    # 3. 예측
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        predicted_prob = probabilities[predicted_class].item() * 100

    # 4. 결과 반환
    result = {
        'predicted_value': predicted_class,
        'prediction_probability': f'{predicted_prob:.2f}%',
        'used_model': model_path
    }
    return result

if __name__ == '__main__':
    # 예시: 'test_image.png'라는 파일을 예측
    # 사용자의 input에 따라 이 부분을 동적으로 변경해야 합니다.
    test_image_path = './sample_images/test_image.png' 
    if os.path.exists(test_image_path):
        result = predict_image(test_image_path)
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print(f"오류: {test_image_path} 파일을 찾을 수 없습니다. 예시 이미지를 준비해 주세요.")