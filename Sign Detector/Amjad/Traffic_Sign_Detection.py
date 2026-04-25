from ultralytics import YOLO

model = YOLO('best.pt')

model.predict(source='test.mov', show=True, save=True)