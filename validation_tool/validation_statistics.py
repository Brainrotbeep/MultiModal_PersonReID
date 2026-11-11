import os
import csv
import json

from config import (

    OUTPUT_DIR,

    ATTRIBUTES

)

# =====================================================
# File Paths
# =====================================================

RESULTS_CSV = os.path.join(

    OUTPUT_DIR,

    "validation_results.csv"

)

SUMMARY_CSV = os.path.join(

    OUTPUT_DIR,

    "validation_summary.csv"

)

REPORT_TXT = os.path.join(

    OUTPUT_DIR,

    "validation_report.txt"

)

STATISTICS_JSON = os.path.join(

    OUTPUT_DIR,

    "validation_statistics.json"

)

LATEX_TABLE = os.path.join(

    OUTPUT_DIR,

    "validation_table.tex"

)


# =====================================================
# Read Validation CSV
# =====================================================

def load_results():

    if not os.path.exists(

        RESULTS_CSV

    ):

        raise FileNotFoundError(

            RESULTS_CSV

        )

    with open(

        RESULTS_CSV,

        newline="",

        encoding="utf-8"

    ) as f:

        reader = csv.DictReader(

            f

        )

        return list(

            reader

        )


# =====================================================
# Compute Statistics
# =====================================================

def compute_statistics(rows):

    attribute_correct = {

        attr: 0

        for attr in ATTRIBUTES

    }

    attribute_total = {

        attr: 0

        for attr in ATTRIBUTES

    }

    dataset_correct = {}

    dataset_total = {}

    overall_correct = 0

    overall_total = 0

    for row in rows:

        dataset = row["Dataset"]

        if dataset not in dataset_correct:

            dataset_correct[dataset] = 0

            dataset_total[dataset] = 0

        for attr in ATTRIBUTES:

            value = int(

                row[attr]

            )

            attribute_total[attr] += 1

            dataset_total[dataset] += 1

            overall_total += 1

            if value == 1:

                attribute_correct[attr] += 1

                dataset_correct[dataset] += 1

                overall_correct += 1

    attribute_accuracy = {}

    for attr in ATTRIBUTES:

        attribute_accuracy[attr] = round(

            100 *

            attribute_correct[attr]

            /

            attribute_total[attr],

            2

        )

    dataset_accuracy = {}

    for ds in dataset_correct:

        dataset_accuracy[ds] = round(

            100 *

            dataset_correct[ds]

            /

            dataset_total[ds],

            2

        )

    overall_accuracy = round(

        100 *

        overall_correct

        /

        overall_total,

        2

    )

    return (

        attribute_accuracy,

        dataset_accuracy,

        overall_accuracy,

        attribute_correct,

        attribute_total

    )


# =====================================================
# Save Summary CSV
# =====================================================

def save_summary(

    dataset_accuracy

):

    with open(

        SUMMARY_CSV,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.writer(

            f

        )

        writer.writerow(

            [

                "Dataset",

                "Accuracy"

            ]

        )

        for ds in dataset_accuracy:

            writer.writerow(

                [

                    ds,

                    dataset_accuracy[ds]

                ]

            )


# =====================================================
# Save JSON
# =====================================================

def save_json(

    attribute_accuracy,

    dataset_accuracy,

    overall

):

    output = {

        "overall_accuracy":

            overall,

        "attribute_accuracy":

            attribute_accuracy,

        "dataset_accuracy":

            dataset_accuracy

    }

    with open(

        STATISTICS_JSON,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            output,

            f,

            indent=4

        )


# =====================================================
# Save Report
# =====================================================

def save_report(

    attribute_accuracy,

    dataset_accuracy,

    overall

):

    with open(

        REPORT_TXT,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(

            "Validation Report\n"

        )

        f.write(

            "=" * 50 + "\n\n"

        )

        f.write(

            f"Overall Accuracy : {overall:.2f}%\n\n"

        )

        f.write(

            "Dataset Accuracy\n"

        )

        f.write(

            "-" * 30 + "\n"

        )

        for ds in dataset_accuracy:

            f.write(

                f"{ds:25s} "

                f"{dataset_accuracy[ds]:.2f}%\n"

            )

        f.write(

            "\nAttribute Accuracy\n"

        )

        f.write(

            "-" * 30 + "\n"

        )

        for attr in attribute_accuracy:

            f.write(

                f"{attr:25s} "

                f"{attribute_accuracy[attr]:.2f}%\n"

            )


# =====================================================
# Save Latex Table
# =====================================================

def save_latex(

    attribute_accuracy,

    overall

):

    with open(

        LATEX_TABLE,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(

r"""\begin{table}[t]
\centering
\caption{Manual validation of generated semantic attributes.}
\label{tab:validation}
\begin{tabular}{lc}
\hline
Attribute & Accuracy (\%)\\
\hline
"""
        )

        names = {

            "age":

                "Age",

            "gender":

                "Gender",

            "upper_color":

                "Upper Clothing Color",

            "upper_type":

                "Upper Clothing Type",

            "lower_color":

                "Lower Clothing Color",

            "lower_type":

                "Lower Clothing Type",

            "footwear_color":

                "Footwear Color"

        }

        for attr in ATTRIBUTES:

            f.write(

                f"{names[attr]} & "

                f"{attribute_accuracy[attr]:.2f}\\\\\n"

            )

        f.write(

f"""\\hline
Overall & {overall:.2f}\\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
        )


# =====================================================
# Main
# =====================================================

def generate_statistics():

    rows = load_results()

    (

        attribute_accuracy,

        dataset_accuracy,

        overall,

        _,

        _

    ) = compute_statistics(

        rows

    )

    save_summary(

        dataset_accuracy

    )

    save_json(

        attribute_accuracy,

        dataset_accuracy,

        overall

    )

    save_report(

        attribute_accuracy,

        dataset_accuracy,

        overall

    )

    save_latex(

        attribute_accuracy,

        overall

    )

    print("\n======================================")

    print("Validation Summary")

    print("======================================")

    print(

        f"Overall Accuracy : "

        f"{overall:.2f}%"

    )

    print("\nDataset Accuracy")

    for ds in dataset_accuracy:

        print(

            f"{ds:25s}"

            f"{dataset_accuracy[ds]:.2f}%"

        )

    print("\nAttribute Accuracy")

    for attr in attribute_accuracy:

        print(

            f"{attr:25s}"

            f"{attribute_accuracy[attr]:.2f}%"

        )

    print("\nResults saved to")

    print(OUTPUT_DIR)


if __name__ == "__main__":

    generate_statistics()