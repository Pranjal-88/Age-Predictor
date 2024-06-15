import cv2
import tensorflow
from tensorflow import keras
from keras.models import load_model
import numpy as np
import random

model=load_model('Models\Age2.h5')

def annotate(img):
    csd=cv2.CascadeClassifier('Datsets\haarcascades\haarcascade_frontalface_default.xml')
    rect=csd.detectMultiScale(img,scaleFactor=1.2,minNeighbors=5)
    for (x,y,w,h) in rect:
        p_img=img[y:y+h,x:w+x]
        p_img=cv2.resize(p_img,(100,100))
        p_img=p_img/255
        p_img=np.expand_dims(p_img,axis=0)
        pred=model.predict(p_img,verbose=3)
        pred=np.argmax(pred,axis=1)
        age=random.randrange(start=pred[0]*5+2,stop=5*(pred[0]+1))
        txt=f'{age} yrs'
        color=(0,255,0)
        cv2.line(pt1=(x,y),pt2=(x+int(0.25*w),y),img=img,color=color,thickness=2)
        cv2.line(pt1=(x+w,y),pt2=(x+w-int(0.25*w),y),img=img,color=color,thickness=2)
        cv2.line(pt1=(x+w,y),pt2=(x+w,y+int(0.25*h)),img=img,color=color,thickness=2)
        cv2.line(pt1=(x+w,y+int(0.75*h)),pt2=(x+w,y+h),img=img,color=color,thickness=2)
        cv2.line(pt1=(x+w,y+h),pt2=(x+int(0.75*w),y+h),img=img,color=color,thickness=2)
        cv2.line(pt1=(x,y+h),pt2=(x+int(0.25*w),y+h),img=img,color=color,thickness=2)
        cv2.line(pt1=(x,y+h),pt2=(x,y+int(0.75*h)),img=img,color=color,thickness=2)
        cv2.line(pt1=(x,y),pt2=(x,y+int(0.25*h)),img=img,color=color,thickness=2)
        cv2.rectangle(img=img,pt1=(x-1,y-16),pt2=(x+1+w,y),color=(100,255,100),thickness=-1)
        cv2.putText(img=img,text=txt,org=(x+3,y-3),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,0,0),thickness=1)
    return img


cam_code=0
cap=cv2.VideoCapture(cam_code)
while True:
    ret,frame=cap.read(0)
    frame=annotate(frame)
    cv2.imshow('Video',frame)
    k=cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()

