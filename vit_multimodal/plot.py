import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("checkpoints/Market-1501/training_loss.csv")

plt.figure(figsize=(6,4))
plt.plot(df["Epoch"], df["Loss"], linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)

plt.savefig(
    "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()