import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

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

# 모델 로드
model = ImprovedCNN()
model.load_state_dict(torch.load('best_mnist_model.pth'))

# 데이터셋 로드 및 합치기
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 1. 사용자 피드백 데이터 로드
retrain_data_path = './retrain_data'
if not os.path.exists(retrain_data_path) or not os.listdir(retrain_data_path):
    print("재학습할 데이터가 없습니다. 재학습을 건너뜁니다.")
    exit()

retrain_dataset = torchvision.datasets.ImageFolder(root=retrain_data_path, transform=transform)

# 2. MNIST 학습 데이터에서 계층적 샘플링
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 각 숫자별로 100개씩 추출
indices = []
targets = mnist_trainset.targets.numpy()
for i in range(10):  # 0부터 9까지
    class_indices = np.where(targets == i)[0]
    np.random.shuffle(class_indices)
    indices.extend(class_indices[:100])

mnist_subset = torch.utils.data.Subset(mnist_trainset, indices)
print(f"MNIST 데이터셋에서 각 숫자별 100개씩 총 {len(mnist_subset)}개 샘플링 완료.")

# 3. 두 데이터셋을 합쳐서 새로운 데이터로더 생성
combined_dataset = torch.utils.data.ConcatDataset([mnist_subset, retrain_dataset])
combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=4, shuffle=True)

# 재학습 진행
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print(f"총 {len(combined_dataset)}개의 데이터로 재학습을 시작합니다...")
for epoch in range(1):  # 1번의 에포크만 추가 학습
    model.train()
    for images, labels in combined_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("재학습이 완료되었습니다.")

# 재학습된 모델 저장
torch.save(model.state_dict(), 'best_mnist_model_retrained.pth')
print("재학습된 모델을 best_mnist_model_retrained.pth로 저장했습니다. 저장 완료")