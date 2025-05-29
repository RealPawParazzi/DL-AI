from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models
from PIL import Image
import torch
import json
import io
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="Test API", description="테스트용 FastAPI 애플리케이션")

# CORS 설정 (앱에서 요청 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 테스트용. 실제 서비스는 도메인 제한하기
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 및 라벨 불러오기
try:
    # 이진 분류 모델 (고양이/강아지)
    catdog_model = models.resnet50(weights=None)
    catdog_model.fc = torch.nn.Linear(catdog_model.fc.in_features, 2)
    catdog_model.load_state_dict(torch.load("model/catdog_resnet50.pth", map_location="cpu"))
    catdog_model.eval()
    catdog_class_names = ['Cat', 'Dog']
except Exception as e:
    print(f"[ERROR] 이진 분류 모델 로딩 실패: {e}")
    catdog_model = None
    catdog_class_names = []

try:
    # 고양이 품종 분류 모델
    cat_model = models.resnet50(weights=None)
    with open("model/catlabels.json", "r", encoding="utf-8") as f:
        cat_label_dict = json.load(f)
        if "labels" in cat_label_dict and isinstance(cat_label_dict["labels"], dict):
            cat_eng2kor = cat_label_dict["labels"]
            cat_idx_to_eng = list(cat_eng2kor.keys())
            cat_idx_to_kor = list(cat_eng2kor.values())
        else:
            raise Exception("catlabels.json 구조 오류")
    cat_model.fc = torch.nn.Linear(cat_model.fc.in_features, len(cat_idx_to_eng))
    cat_model.load_state_dict(torch.load("model/cat_resnet50.pth", map_location="cpu"))
    cat_model.eval()
except Exception as e:
    print(f"[ERROR] 고양이 모델 또는 라벨 파일 로딩 실패: {e}")
    cat_model = None
    cat_eng2kor = {}
    cat_idx_to_eng = []
    cat_idx_to_kor = []

try:
    # 강아지 품종 분류 모델
    dog_model = models.resnet50(weights=None)
    with open("model/labels.json", "r", encoding="utf-8") as f:
        dog_label_dict = json.load(f)
        if "labels" in dog_label_dict and isinstance(dog_label_dict["labels"], dict):
            dog_eng2kor = dog_label_dict["labels"]
            dog_idx_to_eng = list(dog_eng2kor.keys())
            dog_idx_to_kor = list(dog_eng2kor.values())
        else:
            raise Exception("labels.json 구조 오류")
    dog_model.fc = torch.nn.Linear(dog_model.fc.in_features, len(dog_idx_to_eng))
    dog_model.load_state_dict(torch.load("model/dogclassifier.pth", map_location="cpu"))
    dog_model.eval()
except Exception as e:
    print(f"[ERROR] 강아지 모델 또는 라벨 파일 로딩 실패: {e}")
    dog_model = None
    dog_eng2kor = {}
    dog_idx_to_eng = []
    dog_idx_to_kor = []

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 예측 함수
def predict(model, idx_to_eng, idx_to_kor, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        pred_eng = idx_to_eng[pred_idx]
        pred_kor = idx_to_kor[pred_idx]
        confidence = float(probs[pred_idx])
    return pred_eng, pred_kor, confidence

# 데이터 모델 정의
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float

# 임시 데이터 저장소
items = []
item_id_counter = 1

@app.get("/api/")
async def root():
    return {"message": "Welcome to Test API"}

@app.get("/api/items", response_model=List[Item])
async def get_items():
    return items

@app.get("/api/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    item = next((item for item in items if item.id == item_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.post("/api/items", response_model=Item)
async def create_item(item: Item):
    global item_id_counter
    item.id = item_id_counter
    item_id_counter += 1
    items.append(item)
    return item

@app.put("/api/items/{item_id}", response_model=Item)
async def update_item(item_id: int, updated_item: Item):
    item_index = next((index for index, item in enumerate(items) if item.id == item_id), None)
    if item_index is None:
        raise HTTPException(status_code=404, detail="Item not found")
    updated_item.id = item_id
    items[item_index] = updated_item
    return updated_item

@app.delete("/api/items/{item_id}")
async def delete_item(item_id: int):
    item_index = next((index for index, item in enumerate(items) if item.id == item_id), None)
    if item_index is None:
        raise HTTPException(status_code=404, detail="Item not found")
    items.pop(item_index)
    return {"message": "Item deleted successfully"}

@app.post("/api/pet-breed/detect")
async def detect_pet_breed(file: UploadFile = File(...)):
    if catdog_model is None or cat_model is None or dog_model is None:
        raise HTTPException(status_code=500, detail="모델 또는 라벨 파일이 로드되지 않았습니다.")
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # 1. 이진 분류로 고양이/강아지 판별
    with torch.no_grad():
        output = catdog_model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        pet_type = catdog_class_names[pred_idx]
        pet_conf = float(probs[pred_idx])

    # 2. 해당 품종 분류 모델로 품종 추론
    if pet_type == 'Cat':
        breed_eng, breed_kor, breed_conf = predict(cat_model, cat_idx_to_eng, cat_idx_to_kor, input_tensor)
        return {
            "type": "cat",
            "type_confidence": round(pet_conf, 4),
            "breed": breed_kor,
            "breed_en": breed_eng,
            "breed_confidence": round(breed_conf, 4)
        }
    else:
        breed_eng, breed_kor, breed_conf = predict(dog_model, dog_idx_to_eng, dog_idx_to_kor, input_tensor)
        return {
            "type": "dog",
            "type_confidence": round(pet_conf, 4),
            "breed": breed_kor,
            "breed_en": breed_eng,
            "breed_confidence": round(breed_conf, 4)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)