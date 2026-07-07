import json

with open(
    "outputs/features/gallery_image_names.json",
    "r"
) as f:

    names = json.load(f)

print(len(names))
print(names[:5])