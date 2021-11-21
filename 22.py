from PIL import Image
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt


# 定义灰度函数
def rgb1gray(f):
    # 防止三通道相加溢出，所以先转换类型
    f = f.astype(np.float32) / 255

    # 获取图像的三通道
    b = f[:, :, 0]
    g = f[:, :, 1]
    r = f[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # 再把类型转换回来
    gray = (gray * 255).astype('uint8')
    return gray


# 定义腐蚀函数
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


# 定义膨胀函数
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


# 开运算
def wopen(img, B):
    return wdilate(werode(img, B), B)  # 先腐蚀后膨胀


# 闭运算
def wclose(img, B):
    return werode(wdilate(img, B), B)  # 先膨胀后腐蚀


# 击中操作
def whit(img, B):
    W = np.ones((5, 5))
    AB1 = werode(img, B)
    W[1:4, 1:4] = B
    B2 = np.ones((5, 5)) - W
    AB2 = werode(1 - img, B2)
    return AB1 * AB2


# 细化,单一操作
def wthin(img, B):
    return img - whit(img, B)


# 距离变换
def distanceTrans(img):
    A = (1 - img) * 100
    for i in range(0, np.shape(A)[0]):
        for j in range(0, np.shape(A)[1] - 1):
            temp0 = A[i][j]
            temp1 = min(A[i][j - 1] + 3, temp0)
            temp2 = min(A[i - 1][j - 1] + 4, temp1)
            temp3 = min(A[i - 1][j] + 3, temp2)
            temp4 = min(A[i - 1][j + 1] + 4, temp3)
            A[i][j] = temp4

    for i in range(np.shape(A)[0] - 1, -1, 1):
        for j in range(np.shape(A)[1] - 1, -1, 2):
            temp0 = A[i][j]
            temp1 = min(A[i][j + 1] + 3, temp0)
            temp2 = min(A[i + 1][j + 1] + 4, temp1)
            temp3 = min(A[i + 1][j] + 3, temp2)
            temp4 = min(A[i + 1][j - 1] + 4, temp3)
            A[i][j] = temp4
    return A


# 求局部最大值
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


# 定义骨架提取函数
def wextract_sketon(img, B, method='shape'):
    if method == 'shape':  ####形态学骨架提取
        k = 0
        A_kB = 1 - img
        SkA = A_kB - wopen(A_kB, B)
        SA = SkA
        while np.any(A_kB) and k < 10:
            A_kB = werode(A_kB, B)
            SkA = A_kB - wopen(A_kB, B)
            k += 1
            SA = np.clip(SkA + SA, 0, 1)
        return 1 - SA

    elif method == 'distance':
        A_edge = img - werode(img, B)
        A_distance = distanceTrans(A_edge)
        A_lomax = local_max(A_distance)
        A = A_lomax * img
        return A * 255
    return


def find_end(img, K):
    """
    找到端节点
    Parameters:
        img: 输入图像
        K: 结构子序列
    Return:
        只有端节点为前景的图像
    """
    # 像素归一化
    img_ones = img / 255
    img_result = np.zeros_like(img, dtype=np.uint8)

    # 利用结构子序列寻找端点
    for i in K:
        img_temp = np.where(cv2.filter2D(img_ones.copy(), -1, i,
                                         borderType=0) == 3, 1, 0)
        img_result = img_result + img_temp

    img_result *= 255
    return img_result.astype(np.uint8)


# 剪裁算法
def tailor(img):
    """
    裁剪
    Parameters:
        img: 待裁剪图像
    Return:
        裁剪结果图像
    """
    # 生成8个结构子
    k_1 = np.array([[0, 4, 4], [1, 2, 4], [0, 4, 4]], dtype=np.uint8)
    k_2 = np.array([[0, 1, 0], [4, 2, 4], [4, 4, 4]], dtype=np.uint8)
    k_3 = np.array([[4, 4, 0], [4, 1, 2], [4, 4, 0]], dtype=np.uint8)
    k_4 = np.array([[4, 4, 4], [4, 1, 4], [0, 2, 0]], dtype=np.uint8)
    k_5 = np.array([[1, 4, 4], [4, 2, 4], [4, 4, 4]], dtype=np.uint8)
    k_6 = np.array([[4, 4, 1], [4, 2, 4], [4, 4, 4]], dtype=np.uint8)
    k_7 = np.array([[4, 4, 4], [4, 1, 4], [4, 4, 2]], dtype=np.uint8)
    k_8 = np.array([[4, 4, 4], [4, 1, 4], [2, 4, 4]], dtype=np.uint8)

    K = [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8]

    # 细化(去除3个像素组成的分支)
    B = np.ones((3, 3), np.uint8)
    img_thin = wthin(img, B)
    # 找端点
    img_end = find_end(img_thin, K)
    # 膨胀运算,捡回误伤元素
    img_dilate = img_end
    for _ in range(3):
        img_dilate = wdilate(img_dilate,B)
        img_dilate = cv2.bitwise_and(img_dilate, img)
    # 获得裁剪结果
    img_result = cv2.bitwise_or(img_dilate, img_thin)
    return img_result


if __name__ == '__main__':
    img1 = Image.open("780.jpg")
    img1 = np.array(img1)
    # print(img1)
    img2 = rgb1gray(img1)  # 图像灰度化

    # 图像二值化
    avg_gray = np.average(img2)  # 取平均值为阈值
    img3 = np.where(img2[..., :] < avg_gray, 0, 255)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.subplot(2, 3, 1)
    ax = plt.gca()
    plt.axis('off')  # 去掉坐标系
    plt.imshow(img1, cmap='gray')  # 由于Plt显示图像默认三通道，所以开始图像泛绿，需要加参数
    ax.set_title("原图")
    plt.subplot(2, 3, 2)
    ax = plt.gca()
    plt.axis('off')  # 去掉坐标系
    plt.imshow(img2, cmap='gray')
    ax.set_title("灰度图")
    plt.subplot(2, 3, 3)
    ax = plt.gca()
    plt.axis('off')  # 去掉坐标系
    plt.imshow(img3, cmap='gray')
    ax.set_title("二值图")

    plt.subplot(2, 3, 4)
    ax = plt.gca()
    plt.axis('off')  # 去掉坐标系
    B = np.ones((3, 3), np.uint8)
    # img4 = distanceTrans(img3)
    img4 = wextract_sketon(img3 / 255, B, method='shape')
    plt.imshow(img4, cmap='gray')
    ax.set_title("形态学骨架提取")
    plt.subplot(2, 3, 5)
    ax = plt.gca()
    plt.axis('off') #去掉坐标系
    B  = np.array([[0,1,0],[1,1,1],[0,1,0]])
    img5 = wextract_sketon(img3/255,B,method='distance')
    plt.imshow(img5, cmap='gray')
    ax.set_title("距离变换骨架提取")
    plt.subplot(2, 3, 6)
    ax = plt.gca()
    plt.axis('off') #去掉坐标系
    img6 = tailor(img5)
    plt.imshow(img5, cmap='gray')
    ax.set_title("剪裁后的图像")
    plt.show()