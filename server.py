from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import io
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNISTNet(torch.nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)
        self.dropout = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_model(model_path='mnist_model.pth'):
    """Load the trained PyTorch model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    model = MNISTNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

async def predict_from_image(image_data: bytes):
    """Predict digit from image bytes"""
    try:
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            output = MODEL(img_tensor)
            confidence = torch.nn.functional.softmax(output, dim=1)[0] * 100
            predicted_class = output.argmax(dim=1).item()

        logging.info("Prediction successful")
        return predicted_class, confidence[predicted_class].item()

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        contents = await file.read()
        predicted_class, confidence = await predict_from_image(contents)
        
        return JSONResponse({
            "filename": file.filename,
            "predicted_digit": predicted_class,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def serve_index():
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}

if __name__ == "__main__":
    logging.info("Starting FastAPI application...")
    MODEL = load_model()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)