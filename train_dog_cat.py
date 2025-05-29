import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
    # 데이터 경로
    data_dir = './PetImages'

    # 하이퍼파라미터
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0005
    val_ratio = 0.2

    # 전처리 (ResNet50에 맞게)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 데이터셋 분할
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 사전학습된 ResNet50 불러오기
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # 출력층을 2개(고양이/강아지)로 변경
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # 손실함수, 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        train_acc = train_correct / train_total

        # 검증
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), 'model/catdog_resnet50.pth')
    print("모델 저장 완료: model/catdog_resnet50.pth")
