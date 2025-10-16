# src/features.py
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# parameters for LBP
LBP_METHOD = 'uniform'
LBP_P = 8
LBP_R = 1

def color_histogram(image, bins=32, range=(0,256)):
    """Compute color histogram for BGR image (concatenate hist of each channel)."""
    chans = cv2.split(image)
    features = []
    for ch in chans:
        hist = cv2.calcHist([ch], [0], None, [bins], list(range))
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)

def compute_hog(image_gray, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9):
    """Compute HOG on a grayscale image (skimage hog)."""
    # ensure float image in [0,1]
    img = image_gray.astype('float32') / 255.0
    h = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
    return h

def compute_lbp(image_gray, n_points=LBP_P, radius=LBP_R, method=LBP_METHOD, hist_bins=10):
    """Compute LBP histogram."""
    lbp = local_binary_pattern(image_gray, n_points, radius, method)
    # build histogram of patterns
    (hist, _) = np.histogram(lbp.ravel(),
                              bins=np.arange(0, n_points + 3),
                              range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def compute_glcm_props(image_gray, distances=[1], angles=[0], levels=256, props=('contrast','dissimilarity','homogeneity','ASM','energy','correlation')):
    """Compute GLCM and return selected properties averaged across distances/angles."""
    # quantize to reduce levels
    if image_gray.max() > 0:
        img = (image_gray / (image_gray.max() / (levels - 1))).astype('uint8')
    else:
        img = image_gray.astype('uint8')
    glcm = graycomatrix(img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    feats = []
    for p in props:
        try:
            val = graycoprops(glcm, p).mean()
        except Exception:
            val = 0.0
        feats.append(val)
    return np.array(feats)

def extract_features_from_image(image_bgr, resize_to=(256,256), hog_params=None):
    """
    Given a BGR image (as read by cv2), compute concatenated features:
    - color histogram
    - HOG (on grayscale)
    - LBP histogram
    - GLCM properties
    Returns a 1D numpy array of features.
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    # resize
    if resize_to is not None:
        image_bgr = cv2.resize(image_bgr, resize_to)

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # color hist
    ch_feat = color_histogram(image_bgr, bins=32)

    # hog
    if hog_params is None:
        hog_params = dict(pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9)
    hog_feat = compute_hog(image_gray, **hog_params)

    # lbp
    lbp_feat = compute_lbp(image_gray)

    # glcm
    glcm_feat = compute_glcm_props(image_gray, distances=[1,2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64)

    features = np.hstack([ch_feat, hog_feat, lbp_feat, glcm_feat])
    return features
