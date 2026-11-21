
import cv2
import sys

from detection.preprocessing import preprocess_image
from detection.segmentation import (
    create_thresholds,
    create_color_mask,
    fill_capsule_regions,
    find_contours
)
from detection.detector import detect_capsules

from reporting.report import print_report, save_report

from visualization.drawing import draw_detections, display_results


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <image_path>")
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return

    # 1. Preprocessing
    gray, blurred = preprocess_image(image)

    # 2. Segmentation
    threshold_normal, threshold_inverse = create_thresholds(
        blurred
    )

    # Normal thresholding is currently used for detection
    color_mask = create_color_mask(image)
    color_mask = fill_capsule_regions(color_mask)
    contours = find_contours(color_mask)

    cv2.imwrite(
        "images/results/color_mask.png",
        color_mask
    )

    # 3. Detection and filtering
    capsules, report = detect_capsules(contours)

    # 4. Reporting
    print_report(
        report,
        len(contours),
        len(capsules)
    )

    save_report(report, image_path)

    # 5. Visualization
    result = draw_detections(image, capsules)

    display_results(color_mask, threshold_inverse, result)


if __name__ == "__main__":
    main()
