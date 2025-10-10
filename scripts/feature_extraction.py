import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

def load_image(path, size=(128,128)):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Cannot read image {path}")
    img = cv2.resize(img, size)
    return img

def color_hist(img, bins=(8,8,8)):
    hist = cv2.calcHist([img], [0,1,2], None, bins, [0,256,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def glcm_features(gray):
    gray_scaled = (gray / 4).astype('uint8')
    glcm = graycomatrix(gray_scaled, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
    props = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
    feats = [graycoprops(glcm, p)[0,0] for p in props]
    return np.array(feats)

def lbp_features(gray, P=8, R=1):
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins+1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / (gray.shape[0]*gray.shape[1])

def image_stats(gray):
    return [float(np.mean(gray)/255.0), float(np.std(gray)/255.0)]

def hog_features(gray):
    f = hog(gray, orientations=9, pixels_per_cell=(16,16),
            cells_per_block=(1,1), block_norm='L2-Hys', transform_sqrt=True)
    return f

def extract_features(image_path, size=(128,128)):
    img = load_image(image_path, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []
    feats.extend(color_hist(img))
    feats.extend(glcm_features(gray))
    feats.extend(lbp_features(gray))
    feats.append(edge_density(gray))
    feats.extend(image_stats(gray))
    feats.extend(hog_features(gray))
    return np.array(feats, dtype=np.float32)
