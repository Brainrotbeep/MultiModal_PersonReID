import os
import shutil

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# folders to create

folders = [
    "feature_extraction",
    "fusion",
    "outputs/text_embeddings",
    "outputs/fusion"
]

for folder in folders:

    os.makedirs(
        os.path.join(
            BASE_DIR,
            folder
        ),
        exist_ok=True
    )

# files to move

moves = {

    "extract_image_features.py":
        "feature_extraction/extract_image_features.py",

    "fuse_features.py":
        "fusion/fuse_features.py"
}

for src, dst in moves.items():

    src_path = os.path.join(
        BASE_DIR,
        src
    )

    dst_path = os.path.join(
        BASE_DIR,
        dst
    )

    if os.path.exists(src_path):

        shutil.move(
            src_path,
            dst_path
        )

        print(
            f"Moved: {src}"
        )

# new files to create

new_files = [

    "captioning/generate_query_captions.py",
    "captioning/generate_gallery_captions.py",

    "captioning/generate_query_embeddings.py",
    "captioning/generate_gallery_embeddings.py",

    "feature_extraction/extract_query_features.py",
    "feature_extraction/extract_gallery_features.py",

    "fusion/fuse_query_features.py",
    "fusion/fuse_gallery_features.py",

    "Multimodal_final/evaluate_multimodal.py"
]

for file in new_files:

    file_path = os.path.join(
        BASE_DIR,
        file
    )

    if not os.path.exists(file_path):

        open(
            file_path,
            "w",
            encoding="utf-8"
        ).close()

        print(
            f"Created: {file}"
        )

print("\nProject structure updated successfully.")