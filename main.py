import cv2   #include opencv library functions in python
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
model=YOLO("best.pt")
import numpy as np
import numpy
import shutil

shutil.rmtree('runs\detect\predict2')
#---------------------------------------------------------------------------------------------------

def alan(image,kenar):
    mask=np.zeros_like(image)
    mask_color=255
    cv2.fillPoly(mask,kenar,mask_color)
    masked_image= cv2.bitwise_and(image,mask)
    return masked_image

def cizgi_ciz(image,cizgiler):
    image=np.copy(image)
    bos=np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)

    for cizgi in cizgiler:
        for x1,y1,x2,y2 in cizgi:
            cv2.line(bos,(x1,y1),(x2,y2),(255,255,0),thickness=10)

    image=cv2.addWeighted(image,0.8,bos,1,0.0)
    return image

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 200,None,3)
    (thresh, blackAndWhiteImage) = cv2.threshold(canny, 200, 255, cv2.THRESH_BINARY)
    return canny
def surec(image,height,width):
    kenar_bolgesi=[
        (0,height),
        (width/2,height/2),
        (width,height)
    ]
    canny_img= canny(image)
    cropped_img=alan(canny_img,
                     np.array([kenar_bolgesi],np.int32),)
    cizgiler = cv2.HoughLinesP(cropped_img,
                               rho=2,
                               theta=np.pi/180,
                               threshold=100,
                               lines=np.array([]),
                               minLineLength=20,
                               maxLineGap=15)
    cizgili_img= cizgi_ciz(image,cizgiler)
    return cizgili_img

cap = cv2.VideoCapture('videos/bb1.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
	print("Error opening video file")

# Read until video is completed
while(cap.isOpened()):
  
# Capture frame-by-frame
    ret, frame = cap.read()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if ret == True:

        detect=model.predict(source=frame,data="data.yaml",conf=0.25,save=True)
        
        # ar = numpy.asarray(detect)
        image = cv2.imread("runs\detect\predict2\image0.jpg")
        image2=surec(image,height=height,width=width)
        cv2.imshow('Frame', image2)
          
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
  
# Break the loop                                    
    else:
        break

cap.release()

# Closes all the frames
cv2.destroyAllWindows()



