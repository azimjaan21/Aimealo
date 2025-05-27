from ultralytics import YOLO

def train_model():
    model = YOLO('yolo11s.pt') 

    model.train(
        data=r'C:\Users\dalab\Desktop\azimjaan21\CBNU\Big Data\Aimealo\yolo\data.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        device='cuda',  
        patience=10,  # Early stopping
        plots=True,  
        verbose=True  
    )

if __name__ == '__main__':
    train_model()

