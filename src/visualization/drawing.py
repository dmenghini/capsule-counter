
import cv2

from config import DISPLAY_SCALE


def draw_detections(image, capsules):
    result = image.copy()

    for i, contour in enumerate(capsules, start=1):
        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        cv2.circle(
            result,
            (center_x, center_y),
            15,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            result,
            str(i),
            (center_x - 5, center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return result


def display_results(
    threshold_normal,
    threshold_inverse,
    result
):
    normal_display = cv2.resize(
        threshold_normal,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE
    )

    inverse_display = cv2.resize(
        threshold_inverse,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE
    )

    result_display = cv2.resize(
        result,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE
    )

    cv2.imshow("Normal Threshold", normal_display)
    cv2.imshow("Inverse Threshold", inverse_display)
    cv2.imshow("Detected Capsules", result_display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
