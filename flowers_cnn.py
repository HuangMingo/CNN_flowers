import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import pathlib
from sklearn.model_selection import train_test_split

# --- 1. LOAD AND PREPROCESS DATA ---

# Use the Flowers Recognition dataset downloaded from Kaggle.
# Resolve the path from this script so the project can be run from any directory.
project_dir = pathlib.Path(__file__).resolve().parent
data_dir = project_dir / "datasets" / "flowers"

if not data_dir.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found: {data_dir}\n"
        "Download the Kaggle Flowers Recognition dataset and extract it "
        "to datasets/flowers."
    )

# Define class names and their corresponding labels
class_ids = {
    "daisy": 0,
    "dandelion": 1,
    "rose": 2,
    "sunflower": 3,
    "tulip": 4,
}

# Create a dictionary of image paths for each class
class_paths = {
    flower: [
        img_path
        for img_path in (data_dir / flower).iterdir()
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    for flower in class_ids.keys()
}

# Load images and labels into arrays
imgs, labels = [], []
for flower, images in class_paths.items():
    for image in images:
        img = cv2.imread(str(image))
        if img is None:
            print(f"Skipping unreadable image: {image}")
            continue
        # Resize all images to a consistent size
        resized_img = cv2.resize(img, (180, 180))
        imgs.append(resized_img)
        labels.append(class_ids[flower])

print(f"Loaded {len(imgs)} images from {data_dir}")

imgs = np.array(imgs)
labels = np.array(labels)

# Split the data into training and testing sets
train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    imgs, labels, test_size=0.3, random_state=42, stratify=labels
)
val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    temp_imgs, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
)

print("Train:", len(train_imgs))  # 70%
print("Valid:", len(val_imgs))  # 15%
print("Test :", len(test_imgs))  # 15%
# Normalize pixel values from 0-255 to 0-1 for all three splits.
# Converting to float32 also uses less memory than NumPy's default float64.
train_norm = train_imgs.astype("float32") / 255.0
val_norm = val_imgs.astype("float32") / 255.0
test_norm = test_imgs.astype("float32") / 255.0


# --- 2. DEFINE THE CNN MODEL WITH DATA AUGMENTATION ---

# Define data augmentation layers to prevent overfitting
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", input_shape=(180, 180, 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ]
)

# Build the CNN model
model = tf.keras.Sequential(
    [
        data_augmentation,
        # Convolutional Block 1
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        # Convolutional Block 2
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        # Convolutional Block 3
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        # Convolutional Block 4
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        # Convolutional Block 5
        tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        # Global average pooling collapses each feature map to a single value
        # instead of flattening it, which drastically cuts the number of
        # parameters feeding into the Dense layer below and reduces overfitting.
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        # Output layer with units equal to the number of classes
        tf.keras.layers.Dense(len(class_ids)),
    ]
)


# --- 3. COMPILE AND TRAIN THE MODEL ---

# Compile the model
# Using SparseCategoricalCrossentropy because labels are integers, not one-hot encoded
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Print model summary
model.summary()

# Stop training automatically once val_loss stops improving, and keep the
# best-performing weights instead of whatever the last epoch happened to produce.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,  # cho 5 epoch khong cai thien moi dung
    restore_best_weights=True,  # tu dong khoi phuc lai trong so tot nhat
)
# Train the model
history = model.fit(
    train_norm,
    train_labels,
    batch_size=64,
    epochs=50,
    validation_data=(val_norm, val_labels),
    callbacks=[early_stop],
    verbose=1,
)


# --- 4. EVALUATE AND SAVE ---

# Evaluate the model
test_loss, test_acc = model.evaluate(test_norm, test_labels, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Save the model
model.save("flowers_cnn_model.h5")
print("Model saved to flowers_cnn_model.h5")


# --- 5. VISUALIZE RESULTS ---

# Plot and save training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("model_performance.png")
print("Performance plot saved to model_performance.png")
plt.show()

# Visualize predictions
class_names = list(class_ids.keys())
predictions = model.predict(test_norm)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_imgs[i])

    pred_id = np.argmax(predictions[i])
    true_id = test_labels[i]

    pred_class = class_names[pred_id]
    true_class = class_names[true_id]

    color = "blue" if pred_id == true_id else "red"
    plt.xlabel(f"{pred_class} ({true_class})", color=color)

plt.tight_layout()
plt.savefig("predictions.png")
print("Predictions plot saved to predictions.png")
plt.show()
