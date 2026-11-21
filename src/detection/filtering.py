
import cv2

from config import MIN_AREA, MIN_ASPECT_RATIO, MIN_SOLIDITY


def evaluate_contour(contour, index):
    area = cv2.contourArea(contour)

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]

    aspect_ratio = None
    solidity = None

    if width == 0 or height == 0:
        reason = "Invalid rotated rectangle"

    else:
        aspect_ratio = max(width, height) / min(width, height)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            reason = "Invalid convex hull"

        else:
            solidity = area / hull_area

            if area < MIN_AREA:
                reason = "Area below minimum"

            elif aspect_ratio < MIN_ASPECT_RATIO:
                reason = "Aspect ratio below minimum"

            elif solidity < MIN_SOLIDITY:
                reason = "Solidity below minimum"

            else:
                reason = "Accepted"

    accepted = reason == "Accepted"

    report_item = {
        "Object": index,
        "Area": round(area, 2),
        "Aspect Ratio": round(aspect_ratio, 3) if aspect_ratio is not None else "",
        "Solidity": round(solidity, 3) if solidity is not None else "",
        "Result": "POSITIVE" if accepted else "NEGATIVE",
        "Reason": reason
    }

    return accepted, report_item
