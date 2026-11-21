
from detection.filtering import evaluate_contour


def detect_capsules(contours):
    capsules = []
    report = []

    for index, contour in enumerate(contours, start=1):
        accepted, report_item = evaluate_contour(
            contour,
            index
        )

        if accepted:
            capsules.append(contour)

        report.append(report_item)

    return capsules, report
