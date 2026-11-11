import json

with open(
        r"/outputs/captions/Occluded_DukeMTMC/query_caption_od.json",
    encoding="utf-8"
) as f:

    d = json.load(f)

print("Total:", len(d))
print("First 10 keys:")

for i, k in enumerate(d.keys()):
    print(k)
    if i == 9:
        break