from dataset import Market1501, transform

train_path = "data/Market-1501/bounding_box_train"

dataset = Market1501(train_path, transform=transform)

print("Total images:", len(dataset))

img, label, cam = dataset[0]
print("Sample label:", label)
print("Camera:", cam)