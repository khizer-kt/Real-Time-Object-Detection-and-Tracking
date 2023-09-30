# Real-Time-Object-Detection-and-Tracking
This documentation provides an overview and explanation of the code for a real-time object detection and tracking system. The code uses the YOLO (You Only Look Once) object detection model to detect objects in a video stream and tracks them in real-time.
### Prerequisites
Before running the code, ensure that you have the following components set up:

**YOLO Model Files:** You need the YOLO model weights and configuration file. In this example, the code assumes you have the [yolov4.cfg](https://github.com/khizer-kt/Real-Time-Object-Detection-and-Tracking/blob/main/cfg/yolov4.cfg) and [yolov4-tiny.cfg](https://github.com/khizer-kt/Real-Time-Object-Detection-and-Tracking/blob/main/cfg/yolov4-tiny.cfg) files.  
Download the weights files here:  
[yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)  
[yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights)

**COCO Class Names:** The code requires the COCO dataset class names. You should have a file named [coco.names](https://github.com/khizer-kt/Real-Time-Object-Detection-and-Tracking/blob/main/data/coco.names) containing class names.  


**Video Input:**  Provide the path to the video you want to process. You can use a video file or set `cap = cv2.VideoCapture(0)` to use your webcam.  

### Code Explanation  
#### 1. Loading the YOLO Model  
The code starts by loading the YOLO model using OpenCV's `cv2.dnn.readNet` function. It also retrieves the output layer names for later use.
#### 2. Loading COCO Class Names
The code reads the class names from the `coco.names` file and stores them in the classes list.
#### 3. Setting up Video Capture
The code initializes video capture from a video file or webcam. It also resizes the first frame to a fixed size (416x416) for YOLO model input.
#### 4. Object Detection and Tracking Loop
The main part of the code is a loop that continuously reads frames from the video feed, performs object detection, and displays the results.
##### 4.1 Object Detection
- It preprocesses each frame using `cv2.dnn.blobFromImage`.
- Sets the preprocessed frame as the input to the YOLO model.
- Runs forward pass to get detection results.
- Filters detections based on confidence scores (>0.5).
##### 4.2 Non-Maximum Suppression (NMS)
Applies Non-Maximum Suppression to eliminate redundant bounding boxes.
##### 4.3 Drawing Detected Objects
- Draws bounding boxes around detected objects and labels them with class names.
- Random colors are assigned to different classes.
#### 5. Displaying Results
The processed frame with bounding boxes and labels is displayed in a window titled "Real-time Object Detection."
#### 6. Exiting the Program
Press 'Esc' key to exit the program.
#### 7. Release Resources
After the loop ends, the code releases the video capture and closes all OpenCV windows.
### Running the Code
To run the code:

1. Make sure you have the required files and a video source (file or webcam) ready.
2. Ensure you have OpenCV and NumPy installed.
3. Adjust the file paths and video source as needed.
4. Run the code.
### Performance Metrics
The performance of this system can be evaluated based on the following metrics:
- **Detection Accuracy:** How accurately the system detects objects in the video stream.
- **Tracking Accuracy:** The efficiency and accuracy of object tracking as objects move through frames.
- **Real-time Processing Speed (FPS):** Measure the frames per second processed by the system. A higher FPS indicates better real-time performance.
### Challenges and Considerations
- The choice of YOLOv4-tiny model and its configuration was made to balance between speed and accuracy. You can experiment with other YOLO variants or custom-trained models to suit your specific requirements.

- Real-time object tracking can be challenging, especially when objects move quickly or overlap.

- The code uses random colors for object labels, which can be improved for better visualization.

- Ensure you have the necessary dependencies and hardware acceleration (e.g., GPU) for optimal real-time performance.

This codebase serves as a starting point for a real-time object detection and tracking system. Depending on your specific use case, you can fine-tune the model, optimize tracking algorithms, and implement additional features as needed.