import cv2, time, os, tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
	def __init__(self):
		pass


	def readClasses(self, classesFilePath):
		#This method opens the coco.names file in the directory and count the no. of classes in it.

		with open(classesFilePath, 'r') as f:
			self.classesList = f.read().splitlines()

		#Color list for the classes
		self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

		print("Total of " + str(len(self.classesList)) + " classes can be predicted by the model\n")


	def downloadModel(self, modelURL):
		#This method downloads the model from the link in the main.py file.

		fileName = os.path.basename(modelURL)
		self.modelName = fileName[:fileName.index('.')]

		self.cacheDir = "./pretrained_models"
		os.makedirs(self.cacheDir, exist_ok=True)

		get_file(fname=fileName,
			origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)


	def loadModel(self):
		#This method loads the model using tensorflow to make predictions.

		print("Loading Model " + self.modelName)
		tf.keras.backend.clear_session()
		self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
		print("Model " + self.modelName + " Loaded Successfully...\n")


	def createBoundingBox(self, image, threshold = 0.5):
		#This method takes the bounding box coordinates from the predictions made by the model and draw them on the image along with the class name.

		inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
		inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
		inputTensor = inputTensor[tf.newaxis,...]

		detections = self.model(inputTensor)

		bboxs = detections['detection_boxes'][0].numpy()
		classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
		classScores = detections['detection_scores'][0].numpy()

		imH, imW, imC = image.shape

		bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
			iou_threshold=threshold, score_threshold=threshold)

		if len(bboxIdx) != 0:
			for i in bboxIdx:
				bbox = tuple(bboxs[i].tolist())
				classConfidence = round(100*classScores[i])
				classIndex = classIndexes[i]

				classLabelText = self.classesList[classIndex].upper()
				classColor = self.colorList[classIndex]

				displayText = '{}: {}%'.format(classLabelText, classConfidence)

				ymin, xmin, ymax, xmax = bbox

				xmin, xmax, ymin, ymax = (xmin*imW, xmax*imW, ymin*imH, ymax*imH)
				xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

				cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=3)
				cv2.putText(image, displayText, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

		return image


	def predictImage(self, imagePath, threshold=0.5):
		#This method shows the prediction on the image and save it in the current directory.
		
		image = cv2.imread(imagePath)

		bboxImage = self.createBoundingBox(image, threshold)

		os.makedirs("./results", exist_ok=True)
		os.chdir("./results")
		cv2.imwrite(self.modelName + "_result.jpg", bboxImage)
		cv2.imshow("Result", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		print("Prediction made and saved successfully in the results directory.")


	def predictVideo(self, videoPath, threshold=0.5):
		#This method shows the prediction made on a video and the fps
		cap = cv2.VideoCapture(videoPath)

		if (cap.isOpened() == False):
			print("Error opening file...")
			return

		print("Press q to stop the prediction on video")
		(success, image) = cap.read()

		startTime = 1

		while success:
			currentTime = time.time()

			fps = 1/(currentTime - startTime)
			startTime = currentTime

			bboxImage = self.createBoundingBox(image, threshold)

			cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
			cv2.imshow("Result", bboxImage)

			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break

			(success, image) = cap.read()
		cv2.destroyAllWindows()
