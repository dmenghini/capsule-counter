
import csv
from pathlib import Path


FIELDNAMES = [
    "Object",
    "Area",
    "Aspect Ratio",
    "Solidity",
    "Result",
    "Reason"
]


def print_report(report, total_contours, total_capsules):
    print("\n========== DETECTION REPORT ==========")

    for item in report:
        print(
            f"Object {item['Object']}: "
            f"{item['Result']} | "
            f"Area: {item['Area']} | "
            f"Ratio: {item['Aspect Ratio']} | "
            f"Solidity: {item['Solidity']} | "
            f"Reason: {item['Reason']}"
        )

    rejected = total_contours - total_capsules

    print("\n========== SUMMARY ==========")
    print(f"Contours found: {total_contours}")
    print(f"Capsules detected: {total_capsules}")
    print(f"Objects rejected: {rejected}")

    rejection_reasons = {}

    for item in report:
        if item["Result"] == "NEGATIVE":
            reason = item["Reason"]
            rejection_reasons[reason] = (
                rejection_reasons.get(reason, 0) + 1
            )

    if rejection_reasons:
        print("\nRejection reasons:")

        for reason, count in rejection_reasons.items():
            print(f"{reason}: {count}")
    else:
        print("No objects rejected.")


def save_report(report, image_path):
    results_dir = Path("images/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem
    report_path = results_dir / f"{image_name}_report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(report)

    print("\nCSV report saved to:")
    print(report_path)
