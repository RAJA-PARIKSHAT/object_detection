### YoloV5 Inference in OpenCV

- Obtain ONNX models by running the following notebook [code](https://github.com/spmallick/learnopencv/blob/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python/Convert_PyTorch_models.ipynb)
- Make a folder **models** inside the working directory and move the ONNX models inside models directory
- install requirements.txt
- Run inference on image
```
python inference.py -i IMAGEPATH

```
- Run inference on Webcam
```
python inference.py -wb 
```
- For other options 
```
python inference.py --help
```