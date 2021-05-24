import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(grayHist):
    plt.plot(range(256),grayHist,'r',linewidth=1.5,c='red')
    y_maxValue=np.max(grayHist)
    plt.axis([0,255,0,y_maxValue])  #x 和 y的范围
    plt.xlabel("gray level")
    plt.ylabel("number of pixels")
    plt.show()

if __name__=="__main__":
    #转换成灰度图
    img=cv2.imread("color.jpg",0)
    grayHist=cv2.calcHist([img],[0],None,[256],[0,256])

    # 绘制直方图
    plot(grayHist)
    #cv2.imshow("gray_src",img)


    # 直方图均衡化
    equ =cv2.equalizeHist(img)
    res =np.hstack((img,equ))
    cv2.imshow("his_dst", res)

    #限制对比度的自适应图像均衡化
    clahe1=cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl1=clahe1.apply(img)
    res2=np.hstack((img,cl1))
    cv2.imshow("his_clahe_dst",res2)

    cv2.waitKey(0)
