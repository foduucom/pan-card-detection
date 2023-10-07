# pan-card-detection
![](https://github.com/foduucom/pan-card-detection/blob/main/PAN-Card-Detection.jpg)
## Model Overview
The PANCard-Detect model is a yolov8 object detection model trained to detect and locate PAN (Permanent Account Number) cards in images. It is built upon the ultralytics library and fine-tuned using a dataset of annotated PAN card images.

## Intended Use
The model is intended to be used for detecting details like Name,Father Name,DOB,PAN Number, on PAN cards in images. It can be incorporated into applications that require automated detection and extraction of PAN card information from images.

## Performance
The model has been evaluated on a held-out test dataset and achieved the following performance metrics:

Average Precision (AP): 0.90 Precision: 0.92 Recall: 0.89 F1 Score: 0.89 Please note that the actual performance may vary based on the input data distribution and quality.

## Recommendations
Users should be informed about the model's limitations and potential biases. Further testing and validation are advised for specific use cases to evaluate its performance accurately.

Load model and perform prediction:

## How to Get Started with the Model
### To get started with the YOLOv8s object Detection model, follow these steps:
```
pip install ultralyticsplus==0.0.28 ultralytics==8.0.43
```
### Load model and perform prediction:
```
from ultralyticsplus import YOLO, render_result
```
### load model
model = YOLO('foduucom/pan-card-detection')

### set model parameters
```
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image
```
### set image
```
image = '/path/to/your/document/images'
```
### perform inference
```
results = model.predict(image)
```
### observe results
```
print(results[0].boxes)
render = render_result(model=model, image=image, result=results[0])
render.show()
```

## Training Data
The model was trained on a diverse dataset containing images of PAN cards from different sources, resolutions, and lighting conditions. The dataset was annotated with bounding box coordinates to indicate the location of the PAN card within the image.

Total Number of Images: 1,100 Annotation Format: Bounding box coordinates (xmin, ymin, xmax, ymax)

## Fine-tuning Process
* Pretrained Model: TheError: Errors in your YAML metadata model was initialized with a pretrained object detection backbone (e.g. YOLO).
* Loss Function: Mean Average Precision (mAP) loss was used for optimization during training.
* Optimizer: Adam optimizer with a learning rate of 1e-4.
* Batch Size:-1

## Contact 
For inquiries and contributions, please contact us at - [info@foduu.com](mailto:info@foduu.com)
