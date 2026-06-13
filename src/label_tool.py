from pathlib import Path
import cv2
import os
import csv

# path to download photos
CWD = os.getcwd()
IMAGE_DIR = "/data/raw/"
CSV_DIR = "/data"
CSV_FILE = "dataset.csv"

# create the csv file
if not os.path.exists(CSV_DIR+CSV_FILE):
	with open(CSV_FILE, mode="w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["image_path", "x_pixel", "y_pixel", "width", "height"])

# get the images into a list
images = [f for f in os.listdir(CWD+IMAGE_DIR) if f.lower().endswith((".png", ".jpeg", ".jpg"))]

current_image = None
click_coords = None

def click_event(event, x, y, flags, param):
	global click_coords, current_image
	if event==cv2.EVENT_LBUTTONDOWN:
		click_coords = (x, y)
		# draw a visual circle to confim the click location
		cv2.circle(current_image, (x, y), 5, (0, 0, 255), -1)
		cv2.imshow("Labeling", current_image)

WINDOW_NAME = "Labelling."
cv2.destroyAllWindows()
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

for image in images:
	image_path = os.path.join(CWD+IMAGE_DIR, image) 
	current_image = cv2.imread(image_path)
	if current_image is None:
		print(f"Image {current_image} not found.")
		continue
	h, w, _ = current_image.shape
	click_records = None
	# display_image = cv2.resize(current_image, (800, 600))
	cv2.imshow(WINDOW_NAME, current_image)
	cv2.namedWindow(WINDOW_NAME)
	cv2.setMouseCallback(WINDOW_NAME , click_event)

	print(f"Click the ball center for {image} and press 'n' for Next (of 'q' to quit)...")

	while True:
		key = cv2.waitKey(1) & 0xFF
		if key == ord('n') and click_coords is not None:
			with open(CSV_FILE, mode='a', newline='') as f:
				writer = csv.writer(f)
				writer.writerow([image_path, click_coords[0], click_coords[1], w, h])
				break
		elif key == ord('q'):
			cv2.destroyAllWindows()
			exit()
cv2.destroyAllWindows()
print("Labeling complete! Check your dataset.csv file.")
