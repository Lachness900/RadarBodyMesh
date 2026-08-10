
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

methods = [
    "No Buffer",
    "Cumulative",
    "Split Cumulative",
    "32 x 32",
    "64 x 64",
    "128 x 128"
]
accuracy = np.array([
    80.09,
    82.92,
    62.96,
    74.38,
    80.27,
    80.61
])

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(methods))
bars = ax.bar(
    x,
    accuracy,
    width=0.65,
    edgecolor="black",
    linewidth=0.8
)
for bar, value in zip(bars, accuracy):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )
ax.axvline(
    2.5,
    linestyle="--",
    linewidth=1,
    alpha=0.5
)

# Group labels
ax.text(
    1,
    95,
    "Point Cloud",
    ha="center",
    fontsize=13,
    fontweight="bold"
)
ax.text(
    4,
    95,
    "Rasterisation",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=14)
ax.set_title(
    "Effect of Preprocessing Methods",
    fontsize=18,
    fontweight="bold",
    pad=20
)
ax.set_ylim(50, 100)
ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ============================================================
# 2. CLASSICAL MACHINE LEARNING MODELS
# ============================================================

models = [
    "Point-Net",
    "Logistic Regression",
    "Linear SVM",
    "Gaussian NB",
    "KNN (k=15)",
    "Random Forest",
    "KNN (k=10)",
    "MLP Classifier",
    "KNN (k=5)",
    "Decision Tree"
]

model_accuracy = [
    82.92,
    64.39,
    63.79,
    58.50,
    52.86,
    52.78,
    51.84,
    50.90,
    49.62,
    47.82
]

# Sort models from highest to lowest
order = np.argsort(model_accuracy)[::-1]
models_sorted = np.array(models)[order]
accuracy_sorted = np.array(model_accuracy)[order]
plt.figure(figsize=(10, 6))
bars = plt.barh(models_sorted[::-1], accuracy_sorted[::-1])
plt.xlabel("Accuracy (%)")
plt.title("Effect of Model Selection", fontsize=18, fontweight="bold",)
plt.xlim(0, 100)

for bar, accuracy in zip(bars, accuracy_sorted[::-1]):
    plt.text(
        bar.get_width() + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{accuracy:.2f}%",
        va="center",
        fontsize=12,
        fontweight="bold"
    )
plt.tight_layout()
plt.show()


# ============================================================
# 4. EFFECT OF NUMBER OF EPOCHS
# ============================================================

epochs = [
    "30",
    "60",
    "120",
    "240"
]
epoch_accuracy = [
    81.64,
    83.60,
    85.23,
    82.92
]

plt.figure(figsize=(8, 5))
bars = plt.bar(epochs, epoch_accuracy)
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Effect of Training Epochs on Accuracy",fontsize=18, fontweight="bold",)
plt.ylim(50, 100)
for bar, accuracy in zip(bars, epoch_accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )
plt.tight_layout()
plt.show()


# ============================================================
# 5. VALIDATION ACCURACY: 120 VS 240 EPOCHS
# ============================================================

validation_models = [
    "120 Epochs",
    "240 Epochs"
]
validation_accuracy = [
    79.61,
    79.85
]

plt.figure(figsize=(7, 5))
bars = plt.bar(validation_models, validation_accuracy)
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy: 120 vs 240 Epochs",fontsize=18, fontweight="bold",)
plt.ylim(50, 100)
for bar, accuracy in zip(bars, validation_accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )
plt.tight_layout()
plt.show()