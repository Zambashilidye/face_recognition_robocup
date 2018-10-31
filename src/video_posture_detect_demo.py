import cv2 



#img = cv2.imread(filepath) # 读取图片 
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换灰色 


HOGCascade = cv2.HOGDescriptor()
HOGCascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
color = (0, 255, 0) # 定义绘制颜色 
# starting video streaming
video_capture = cv2.VideoCapture(0)
while True:
    img = video_capture.read()[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换灰色 
    # 调用识别 
    winStride = (8,8)
    padding = (16,16)
    scale = 1.03
    meanshift = 1
    (rects, weights) = HOGCascade.detectMultiScale(gray, winStride=winStride,
                                            padding=padding,
                                            scale=scale,
                                            useMeanshiftGrouping=meanshift)

    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,200,255), 2)
        print(x,y,w,h)
    cv2.imshow("image", img) # 显示图像 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()