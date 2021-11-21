""" 编一个程序实现如下功能：
1、	读入一幅指纹图像（自己找）；
2、	对图像进行二值化（方法自定，可以是阈值法）；
2、采用形态学骨架提取和距离变换骨架提取两种算法分别提取图像骨架；
3、采用裁剪算法，并分析其效果。 """

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import cv2


def binary(img, T=125, Tmin=10, time_max=1000):
    Gh = 255
    Gl = 0
    t = 0
    img_L = np.array([i for j in img for i in j])
    img_bin = img
    T0 = 0
    while abs(T - T0) >= Tmin and t <= time_max:
        T0 = T
        Gh = np.sum(img_L[img_L > T]) / (np.sum(img_L > T))
        Gl = np.sum(img_L[img_L <= T]) / (np.sum(img_L <= T))
        T = (Gh + Gl) / 2
        t = t + 1
    np.putmask(img_bin, img_bin >= T, 255)
    np.putmask(img_bin, img_bin < T, 0)
    img_bin = 255 - img_bin
    return img_bin


#####腐蚀#######
def werode(f, B):
    L, h = f.shape
    wW, hW = B.shape
    wp, hp = wW // 2, hW // 2
    SB = np.sum(B)
    """ padding """
    row_pad_up = np.zeros((wp, h))
    row_pad_down = np.zeros((wp, h))
    col_pad_left = np.zeros((L + 2 * wp, hp))
    col_pad_right = np.zeros((L + 2 * wp, hp))  ##扩充的行列块初始化全为零
    img_p = np.row_stack((row_pad_up, f, row_pad_down))
    img_p = np.column_stack((col_pad_left, img_p, col_pad_right))
    """ 卷积操作 """
    out_img = np.zeros((L, h))  # 输出图像初始化
    w_img = np.zeros((wW, hW))  # 定义一个原始图像中与卷积核对应矩阵
    W = B
    for i in range(L):
        for j in range(h):
            w_img = img_p[i:i + wW, j:j + hW]
            if SB == np.sum(W * w_img):
                out_img[i, j] = 1
    out_img = np.clip(out_img, 0, 255)
    return out_img


##############膨胀###################
def wdilate(f, B):
    L, h = f.shape
    wW, hW = B.shape
    wp, hp = wW // 2, hW // 2
    """ padding """
    row_pad_up = np.zeros((wp, h))
    row_pad_down = np.zeros((wp, h))
    col_pad_left = np.zeros((L + 2 * wp, hp))
    col_pad_right = np.zeros((L + 2 * wp, hp))  ##扩充的行列块初始化全为零
    img_p = np.row_stack((row_pad_up, f, row_pad_down))
    img_p = np.column_stack((col_pad_left, img_p, col_pad_right))
    """ 卷积操作 """
    out_img = np.zeros((L, h))  # 输出图像初始化
    w_img = np.zeros((wW, hW))  # 定义一个原始图像中与卷积核对应矩阵
    W = B
    for i in range(L):
        for j in range(h):
            w_img = img_p[i:i + wW, j:j + hW]
            if np.sum(W * w_img) > 0:
                out_img[i, j] = 1
    out_img = np.clip(out_img, 0, 255)
    return out_img


###########开运算###################
def wopen(img, B):
    return wdilate(werode(img, B), B)


###############闭运算###############
def wclose(img, B):
    return werode(wdilate(img, B), B)


################击中操作#################
def whit(img, B):
    W = np.ones((5, 5))
    AB1 = werode(img, B)
    W[1:4, 1:4] = B
    B2 = np.ones((5, 5)) - W
    AB2 = werode(1 - img, B2)
    return AB1 * AB2


def hit(img, B):
    W = np.ones((5, 5), np.uint8)
    AB1 = cv2.erode(img, B)
    W[1:4, 1:4] = B
    B2 = np.ones((5, 5), np.uint8) - W
    AB2 = cv2.erode(1 - img, B2)
    return AB1 * AB2


##############细化,单一操作#################
def wthin(img, B):
    return img - hit(img, B)


##############细化过程##########################
def wthinp(A, Bs):
    AthinB = A
    for i in range(3):
        for B in Bs:
            AthinB = wthin(AthinB, B)
    return AthinB


###############距离变换#################
def distanceTrans(img):
    A = (1 - img) * 100
    for i in range(0, np.shape(A)[0]):
        for j in range(0, np.shape(A)[1] - 1):
            temp0 = A[i][j]
            temp1 = min(A[i][j - 1] + 1, temp0)
            temp2 = min(A[i - 1][j - 1] + 2, temp1)
            temp3 = min(A[i - 1][j] + 1, temp2)
            temp4 = min(A[i - 1][j + 1] + 2, temp3)
            A[i][j] = temp4

    for i in range(np.shape(A)[0] - 2, 0, -1):
        for j in range(np.shape(A)[1] - 2, 0, -1):
            temp0 = A[i][j]
            temp1 = min(A[i][j + 1] + 1, temp0)
            temp2 = min(A[i + 1][j + 1] + 2, temp1)
            temp3 = min(A[i + 1][j] + 1, temp2)
            temp4 = min(A[i + 1][j - 1] + 2, temp3)
            A[i][j] = temp4
    return A


#####################求局部最大值#################
def local_max(img):
    img_lm = img
    l, h = img.shape
    t = 1
    for i in range(t, l + 1 - t):
        for j in range(t, h + 1 - t):
            if img_lm[i, j] == np.max(img_lm[i - t:i + t, j - t:j + t]):
                img_lm[i, j] = 1
    np.putmask(img_lm, img_lm > 1, 0)
    return img_lm


################骨架提取,全是自己写的############
def wextract_sketon(img, B, method='shape'):
    if method == 'shape':  ####形态学骨架提取
        k = 0
        A_kB = img
        SkA = A_kB - wopen(A_kB, B)
        SA = SkA
        while np.any(A_kB) and k < 20:
            A_kB = werode(A_kB, B)
            SkA = A_kB - wopen(A_kB, B)
            k += 1
            SA = np.clip(SkA + SA, 0, 1)
            plt.imshow(SA, cmap='gray')
            plt.pause(1)
            plt.close()
        print(k)
        return SA
    elif method == 'distance':
        print('distance')
        A_edge = img - cv2.erode(img, B)
        A_distance = distanceTrans(A_edge)
        A_lomax = local_max(A_distance)
        A = A_lomax * img
        return A * 255
    return


# ################骨架提取,其他的东西是调用的############
# def extract_sketon(img,B,method='shape'):
#     if method == 'shape': ####形态学骨架提取
#         k = 0
#         A_kB = img
#         SkA = A_kB - cv2.morphologyEx(A_kB,cv2.MORPH_OPEN,B)
#         SA = SkA
#         while np.any(A_kB) and k<20:
#             A_kB = cv2.erode(A_kB,B)
#             SkA = A_kB - cv2.morphologyEx(A_kB,cv2.MORPH_OPEN,B)
#             k += 1
#             SA = np.clip(SkA+SA,0,1)
#             # plt.imshow(SA,cmap='gray')
#             # plt.pause(1)
#             # plt.close()
#         print(k)
#         return SA
#     elif method == 'distance':
#         print('distance')
#         A_edge = img - cv2.erode(img,B)
#         A_distance = distanceTrans(A_edge)
#         A_lomax = local_max(A_distance)
#         A = A_lomax*img
#         return A*255
#     elif method == 'distance1':
#         A = img
#         t = 0
#         while np.any(A) and t<5:
#             print(t)
#             At = A
#             A_edge = cv2.erode(A,B)
#             A_distance = distanceTrans(A_edge)
#             A = A_distance
#             np.putmask(A,A>1,0)
#             t += 1
#         return At*255
#     return
# #####################剪裁算法####################
# def cut(A):
#     X1 = wthinp(A,Bs)
#     X2 = A
#     X2 = 0
#     H = np.ones((3,3))
#     for B in Bs:
#         print('3')
#         X2 = X2+hit(X1,B)
#     np.clip(X2,0,1)
#     X3 = cv2.dilate(X2,H)*A
#     X4 = np.clip(X3+X1,0,1)
#     return X4

if __name__ == "__main__":
    img = Image.open("780.jpg")
    img = np.array(img)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.title('指纹原图')
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('灰度直方图')

    plt.figure(2)
    plt.subplot(1, 2, 1)
    img_bin = binary(img, Tmin=0.01)
    plt.imshow(img_bin, cmap="gray")
    plt.title('二值化')
    plt.subplot(1, 2, 2)
    plt.hist(img_bin.ravel(), 256, [0, 256])
    plt.title('二值化后灰度直方图')

    plt.figure(3)
    B = np.ones((3, 3), np.uint8)
    img_wskelon_shape = wextract_sketon(img_bin / 255, B, method='shape')
    plt.imshow(img_wskelon_shape, cmap="gray")
    plt.title('形态学骨架提取')

    # plt.figure(4)
    # plt.title('距离变换骨架提取')
    # #skelon = morphology.skeletonize(img_bin/255)
    # img_wskelon_distance = wextract_sketon(img_bin/255,B,method='distance')
    # plt.imshow(img_wskelon_distance,cmap="gray")

    # plt.figure(5)
    # img_cut=cut(img_wskelon_shape)
    # plt.title('剪裁')
    # plt.imshow(img_cut,cmap='gray')
    plt.show()

    # imgdis = skimage.morphology.medial_axis(img_bin, mask=None, return_distance=False)
    # plt.imshow(imgdis,cmap='gray')
    # plt.show()