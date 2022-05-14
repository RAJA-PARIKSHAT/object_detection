import cv2
import os 
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument("-i", "--IMAGEPATH", required = False, help = "Path to a single input_image")
args.add_argument("-wb", "--WEBCAM", required = False, action= 'store_true',  help = "To run inference on webcam")
args.add_argument("-id", "--IMAGEDIRECTORY", required = False, help = "Path to input_images directory, inference will be done on input_images one by one")
args.add_argument("-v", "--VIDEOPATH", required = False, help = "Path to a video file")
args.add_argument("-m", "--MODEL", required = False, default= "models/yolov5s.onnx", help = "Path to ONNX model, Default YOLOv5s will be loaded")
args.add_argument("-cl", "--CLASSES", required = False, default= "coco.names", help = "Path to classes.names file. Default coco classes will be loaded")
args.add_argument("-iW", "--WIDTH", required = False, default = 640, help = "Required input_image width")
args.add_argument("-iH", "--HEIGHT", required = False, default = 640, help = "Required input_image height")
args.add_argument("-nT", "--NMSTHRESH", required = False,default = 0.45, help = "NMS threshold")
args.add_argument("-cT", "--CNFTHRESH", required = False, default = 0.45, help = "confidence threshold")

arguments = args.parse_args()
options = vars(arguments)

def forward(input_image, net):

    input_blob = cv2.dnn.blobFromImage(image = input_image, scalefactor = 1/255., size = (INP_WIDTH, INP_HEIGHT), mean = [0,0,0], swapRB = 1, crop = False)

    net.setInput(input_blob)
    output_layers = net.getUnconnectedOutLayersNames()

    outputs = net.forward(output_layers)

    return outputs


def post_process_image(input_image, outputs):
    

    ORIGINAL_HEIGHT, ORIGINAL_WIDTH = input_image.shape[:2]

    x_factor = ORIGINAL_WIDTH / INP_WIDTH
    y_factor = ORIGINAL_HEIGHT / INP_HEIGHT

    boxes = []
    classIDS = []
    confidences = []

    for out in outputs:

        out = out[0]

        for detection in out:

            det_confidence = detection[4]
            
            if det_confidence > confThreshold:

                class_scores = detection[5:]
                classID      = np.argmax(class_scores)

                center_x = detection[0]
                center_y = detection[1]
                width    = detection[2]
                height   = detection[3]

                left = int((center_x - width/2)*x_factor)
                top  = int((center_y - height/2)*y_factor)
                
                classIDS.append(classID)
                confidences.append(det_confidence)
                boxes.append([left, top, int(width*x_factor), int(height * y_factor)])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for index in indices:
        
        box = boxes[index]
        ID  = classIDS[index]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = left + width
        bottom = top + height
        color = class_colors[ID]
        cv2.rectangle(input_image, (left, top), (right, bottom), (25,int(color[0]),int(color[1])), 2)
        label = "{}:{:.2f}".format(classes[ID], confidences[index])

        cv2.putText(input_image, label, (left, top - 15), cv2.FONT_HERSHEY_COMPLEX, 0.4,(25,int(color[0]),int(color[1])), 1)
    
    return input_image

def inference():
    if arguments.IMAGEPATH:

        image = cv2.imread(options["IMAGEPATH"])
        outs = forward(image, net= net)
        output_image = post_process_image(image, outputs= outs)
        cv2.imshow('image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    elif arguments.IMAGEDIRECTORY:

        root = options["IMAGEDIRECTORY"]
        image_paths = os.listdir(root)

        for image_path in image_paths:
            
            image = cv2.imread(os.path.join(root, image_path))
            outs = forward(image, net= net)
            output_image = post_process_image(image, outputs= outs)
            cv2.imshow('image', output_image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        return

    elif arguments.WEBCAM:
        camera = cv2.VideoCapture(0)

        while True:

            ret, frame = camera.read()
            if ret:

                outs = forward(frame, net)
                output_image = post_process_image(frame, outs)
                cv2.imshow('stream', output_image)

                if cv2.waitKey(1) == ord('q'):
                    break

        camera.release()
        cv2.destroyAllWindows()
        return

    elif arguments.VIDEOPATH:

        video = cv2.VideoCapture(options["VIDEOPATH"])

        while (video.isOpened()):

            ret, frame = video.read()

            if ret:
                outs = forward(frame, net)
                output_image = post_process_image(frame, outs)
                cv2.imshow('stream', output_image)

                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
        video.release()
        cv2.destroyAllWindows()
        return







if __name__ == '__main__':

    INP_HEIGHT = int(options["HEIGHT"])
    INP_WIDTH  = int(options["WIDTH"])
    
    confThreshold = float(options["CNFTHRESH"])
    nmsThreshold  = float(options["NMSTHRESH"])

    classes_path = options["CLASSES"]
    model_path   = options["MODEL"]
    
    classes = None

    with open(classes_path, "rt") as file:
        classes = file.read().rstrip("\n").split("\n")
    
    class_colors = np.random.randint(0,256, size = (len(classes), 2),dtype= np.uint8)
    net = cv2.dnn.readNetFromONNX(model_path)

    inference()