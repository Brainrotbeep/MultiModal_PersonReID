import os
import csv
import cv2
import numpy as np

from config import (
    OUTPUT_DIR,
    ATTRIBUTES
)

from utils import (
    load_all_samples,
    load_image,
    resize_image,
    build_caption_text
)

from validation_statistics import generate_statistics

# =====================================================
# Output CSV
# =====================================================

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "validation_results.csv"
)

# =====================================================
# CSV Header
# =====================================================

CSV_HEADER = [

    "Dataset",

    "Image"

] + ATTRIBUTES

# =====================================================
# Create CSV
# =====================================================

def initialize_csv():

    if os.path.exists(
        RESULTS_CSV
    ):
        return

    with open(

        RESULTS_CSV,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            CSV_HEADER
        )

# =====================================================
# Resume Support
# =====================================================

def load_completed_images():

    completed = set()

    if not os.path.exists(
        RESULTS_CSV
    ):
        return completed

    with open(

        RESULTS_CSV,

        newline="",

        encoding="utf-8"

    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            completed.add(

                row["Image"]

            )

    return completed

# =====================================================
# Save One Validation
# =====================================================

def save_result(

    dataset,

    image_name,

    results

):

    row = [

        dataset,

        image_name

    ]

    row.extend(

        results

    )

    with open(

        RESULTS_CSV,

        "a",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            row
        )
# =====================================================
# Create Text Panel
# =====================================================

def create_text_panel(

    sample,

    index,

    total

):

    panel = np.ones(

        (

            720,

            500,

            3

        ),

        dtype=np.uint8

    ) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 40

    cv2.putText(

        panel,

        f"Image {index}/{total}",

        (20, y),

        font,

        0.8,

        (0, 0, 255),

        2

    )

    y += 45

    cv2.putText(

        panel,

        f"Dataset : {sample['dataset']}",

        (20, y),

        font,

        0.6,

        (0, 0, 0),

        2

    )

    y += 35

    cv2.putText(

        panel,

        os.path.basename(

            sample["image_path"]

        ),

        (20, y),

        font,

        0.55,

        (80, 80, 80),

        1

    )

    y += 45

    cv2.line(

        panel,

        (10, y),

        (470, y),

        (0, 0, 0),

        1

    )

    y += 35

    lines = build_caption_text(

        sample["attributes"]

    )

    for line in lines:

        cv2.putText(

            panel,

            line,

            (20, y),

            font,

            0.60,

            (0, 0, 0),

            2

        )

        y += 38

    y += 20

    cv2.line(

        panel,

        (10, y),

        (470, y),

        (0, 0, 0),

        1

    )

    y += 40

    instructions = [

        "Incorrect Fields",

        "",

        "1  Age",

        "2  Gender",

        "3  Upper Color",

        "4  Upper Type",

        "5  Lower Color",

        "6  Lower Type",

        "7  Footwear Color",

        "",

        "ENTER : All Correct",

        "SPACE : Next Image",

        "ESC : Exit"

    ]

    for text in instructions:

        cv2.putText(

            panel,

            text,

            (20, y),

            font,

            0.55,

            (120, 0, 0),

            1

        )

        y += 28

    return panel


# =====================================================
# Display Window
# =====================================================

def show_sample(

    sample,

    index,

    total

):
    print(sample["image_path"])

    image = load_image(sample["image_path"])

    print(image.shape)

    image = load_image(

        sample["image_path"]

    )

    image = resize_image(

        image

    )

    panel = create_text_panel(

        sample,

        index,

        total

    )

    h = max(

        image.shape[0],

        panel.shape[0]

    )

    canvas = np.ones(

        (

            h,

            image.shape[1] + panel.shape[1],

            3

        ),

        dtype=np.uint8

    ) * 255

    canvas[

        :image.shape[0],

        :image.shape[1]

    ] = image

    canvas[

        :panel.shape[0],

        image.shape[1]:

    ] = panel

    cv2.imshow(
        "Attribute Validation",
        canvas
    )

    cv2.waitKey(1)
# =====================================================
# Validate One Sample
# =====================================================

def validate_sample(

    sample,

    index,

    total

):

    show_sample(

        sample,

        index,

        total

    )

    print("\n" + "=" * 60)

    print(

        f"Image {index}/{total}"

    )

    print(

        f"Dataset : {sample['dataset']}"

    )

    print(

        os.path.basename(

            sample["image_path"]

        )

    )

    print("=" * 60)

    print("\nGenerated Attributes\n")

    for i, attr in enumerate(

        ATTRIBUTES,

        start=1

    ):

        value = sample[

            "attributes"

        ].get(

            attr,

            "Unknown"

        )

        print(

            f"{i}. {attr:15s}: {value}"

        )

    print("\n")

    print(

        "Press ENTER if everything is correct."

    )

    print(

        "Otherwise type incorrect field numbers."

    )

    print(

        "Example : 3 5 7"

    )

    print(

        "Type q to quit."

    )

    user_input = input(

        "\n>> "

    ).strip()

    if user_input.lower() == "q":

        cv2.destroyAllWindows()

        print(

            "\nValidation Interrupted."

        )

        exit()

    results = [

        1

    ] * len(

        ATTRIBUTES

    )

    if user_input != "":

        try:

            incorrect = [

                int(x)

                for x in user_input.split()

            ]

            for idx in incorrect:

                if 1 <= idx <= len(

                    ATTRIBUTES

                ):

                    results[

                        idx - 1

                    ] = 0

        except Exception:

            print(

                "\nInvalid Input."

            )

            print(

                "Assuming all attributes are correct."

            )

    save_result(

        sample["dataset"],

        os.path.basename(

            sample["image_path"]

        ),

        results

    )

    cv2.destroyAllWindows()

    return results

# =====================================================
# Main
# =====================================================

def main():

    print("\n========================================")
    print("Semantic Attribute Validation Tool")
    print("========================================\n")

    initialize_csv()

    completed = load_completed_images()

    samples = load_all_samples()

    total = len(samples)

    print(f"Total Sampled Images : {total}")

    print(
        f"Already Validated    : {len(completed)}"
    )

    print()

    current = len(completed)

    for sample in samples:

        image_name = os.path.basename(
            sample["image_path"]
        )

        if image_name in completed:

            continue

        current += 1

        validate_sample(

            sample,

            current,

            total

        )

    print("\n========================================")
    print("Validation Completed")
    print("========================================\n")

    cv2.destroyAllWindows()

    generate_statistics()

    print("\n========================================")
    print("All Reports Generated Successfully")
    print("========================================")

    print(
        f"\nResults Folder:\n{OUTPUT_DIR}"
    )


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":

    main()