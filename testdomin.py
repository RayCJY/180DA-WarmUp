import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    height, width, _ = frame.shape
    top_left_y = height // 2 - 20
    bottom_right_y = height // 2 + 20
    top_left_x = width // 2 - 20
    bottom_right_x = width // 2 + 20

    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    
    clt = KMeans(n_clusters=3)
    clt.fit(img)
    
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    normalizedRGB = clt.cluster_centers_.astype('uint8')
    hsvCenters = cv2.cvtColor(np.array([normalizedRGB]), cv2.COLOR_RGB2HSV)
    print("Printing RGB values of the 3 cluster centers:\n", clt.cluster_centers_, "\nPrinting their HSV values:\n", hsvCenters)

    
    plt.axis("off")
    plt.imshow(bar)
    plt.pause(0.05) 
    plt.clf()

    
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Cropped Region', roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()
