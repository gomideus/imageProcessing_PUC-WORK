import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy
import time
from PIL import Image
from skimage.measure import shannon_entropy

def calcularCaracteristicas(imgPath):

    grayImg = Image.open(imgPath).quantize(32)

    start_time = time.time()

    distances = [1, 2, 4, 8, 16]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['energy', 'homogeneity']

    glcm = graycomatrix(grayImg, 
                        distances=distances, 
                        angles=angles,
                        levels=None,
                        symmetric=True,
                        normed=True)

    feats = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
    end = time.time() - start_time
    
    return feats, shannon_entropy(grayImg), end
