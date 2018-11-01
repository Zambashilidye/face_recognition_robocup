import cv2  
  
img = cv2.imread('logo.jpg')  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
  
img, contours, hierarchy= cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
cv2.drawContours(img,contours,-1,(0,0,255),3)  
x,y,w,h = cv2.boundingRect(contours[0])
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.imshow("img", img)  
contours[0]
print(type(contours[0]))
#cv2.waitKey(10)