#mostly taken from https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar



cap = cv2.VideoCapture(0)
clt = KMeans(n_clusters=3) #cluster number
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
while True:
    _, imgOriginal = cap.read()
    croppedImg = imgOriginal[len(imgOriginal)//2 - 20 : len(imgOriginal)//2 + 20, len(imgOriginal[1])//2 -20 : len(imgOriginal[1])//2 + 20]
    img = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2RGB)
    

    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    
    clt.fit(img)
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    normalizedRGB = clt.cluster_centers_.astype('uint8')
    #normalizedRGB = [[int(e) for e in row] for row in clt.cluster_centers_]
    hsvCenters = cv2.cvtColor(np.array([normalizedRGB]), cv2.COLOR_RGB2HSV)
    print("Printing RGB values of the 3 cluster centers:\n", clt.cluster_centers_, "\nPrinting their HSV values:\n", hsvCenters)

    plt.axis("off")
    plt.imshow(bar)
    cv2.imshow('img',imgOriginal)
    cv2.imshow('cropped', croppedImg)
    plt.show()
    