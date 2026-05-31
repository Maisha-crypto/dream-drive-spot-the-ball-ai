import cv2
import os
import csv

# path to download photos
IMAGE_DIR = "/data/raw/"
CSV_FILE = "dataset.csv"

# create the csv file
if not.os.path.exist(CSV_FILE):
  with open(CSV_FILE, mode="w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["image_path", "x_pixel", "y_pixel", "width", "height"])
# get the images into a list
images = [f for f in os.listdir(IMAGE_DIR) if f.lower().ends with ((".png", ".jpeg", ".jpg"))]

current_image = None
click_coords = None

def click_event(event, x, y, flags, param):
	global click_coords, current_image
	if event==cv2.EVENT_LBUTTONDOWN:
		click_coords = (x, y)
		# draw a visual circle to confim the click location
		image_copy = current_image.copy()
		cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
		cv2.imshow("Labeling", img_copy)

for image in images:
	image_path = os.path.join(IMAGE_DIR, image_name) 
	current_image = cv2.imread(img_path)
	if current_image is None:
		print(f"Image {current_image} not found.")
		continue
h, w, _ = current_image.shape
click_records = None

cv2.imshow("Labeling", current_image)
cv2.setMouseCallback("Labelling" , click_event)

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
