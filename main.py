from detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
imagePath = "test/1.jpg"
videoPath = "test/ris.mp4" #make this variable equal 0 if you want to make predictions on video input via webcam
threshold = 0.5
classFile = "coco.names"

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)