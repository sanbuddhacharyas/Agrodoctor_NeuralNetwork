import cv2
import glob

data = glob.glob('color/leaf_images/*.JPG')
print(data)
i=0
for d in data:
    read = cv2.imread(d,cv2.IMREAD_COLOR)
    cv2.imwrite(str(i)+".jpg", read)
    i += 1
