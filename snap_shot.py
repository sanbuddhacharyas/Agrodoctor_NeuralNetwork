import cv2

Left_camera = cv2.VideoCapture(1)
Left_camera.set(cv2.CAP_PROP_FRAME_WIDTH,1296 )
Left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#Right_camera = cv2.VideoCapture(1)
i=0
while(1):
    ret1,img1 = Left_camera.read(cv2.IMREAD_COLOR)
   # ret2,img2 = Right_camera.read(cv2.IMREAD_COLOR)
   
    cv2.imshow('Left',img1)
   # cv2.imshow('Right',img2)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == 32:
        string = 'Leaf/Leaf'+str(i)+'.jpg'
        cv2.imwrite(string,img1)
        i += 1