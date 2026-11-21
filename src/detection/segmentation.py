
import cv2
import numpy as np


def create_thresholds(blurred):
    _, threshold_normal = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, threshold_inverse = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return threshold_normal, threshold_inverse







def create_color_mask(image, min_component_area=100):
    """
    Enhance local contrast using CLAHE and create
    a foreground mask based on image brightness.
    """

    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Separate lightness and color information
    lightness, a, b = cv2.split(lab)

    # Enhance local contrast in the lightness channel
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_lightness = clahe.apply(lightness)

    # Smooth small variations
    enhanced_lightness = cv2.GaussianBlur(
        enhanced_lightness,
        (5, 5),
        0
    )

    # Adaptive thresholding
    color_mask = cv2.adaptiveThreshold(
        enhanced_lightness,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        -5
    )

    # Remove small isolated regions
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
    color_mask,
    connectivity=8
    )

    # Determine which components are large enough
    keep = stats[:, cv2.CC_STAT_AREA] >= min_component_area

    # Never include the background (component 0)
    keep[0] = False

    # Build the cleaned mask in one vectorized operation
    cleaned_mask = (keep[labels] * 255).astype(np.uint8)

    return cleaned_mask



def fill_capsule_regions(mask):
    """
    Fill enclosed dark regions inside white capsule boundaries.
    """

    # Find external contours of the current white regions
    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(mask)

    # Fill the enclosed regions
    cv2.drawContours(
        filled_mask,
        contours,
        -1,
        255,
        thickness=cv2.FILLED
    )

    return filled_mask



def find_contours(threshold):
    contours, _ = cv2.findContours(
        threshold.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours
