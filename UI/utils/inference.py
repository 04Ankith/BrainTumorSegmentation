import os
import cv2
import uuid
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model

# Load the model (already trained and saved)
model = load_model("models/unet_model_f.h5", compile=False)

# Create output folder if not exists
os.makedirs("static/outputs", exist_ok=True)

def normalize(img):
    img = np.nan_to_num(img)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

def predict_and_return_images(nifti_path, slice_index=80, image_size=(128,128)):
    # Load and normalize NIfTI image
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()
    img_slice = normalize(volume[:, :, slice_index])

    # Resize to model input
    input_resized = cv2.resize(img_slice, image_size)
    input_tensor = input_resized[np.newaxis, ..., np.newaxis]

    # Predict
    pred = model.predict(input_tensor)[0]  # (128,128,4)
    pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)  # (128,128)

    # Resize everything back to original size
    H, W = img_slice.shape
    input_display = cv2.resize(input_resized, (W, H))
    mask_display = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Overlay creation
    input_rgb = np.stack([input_display]*3, axis=-1)
    input_rgb = (input_rgb * 255).astype(np.uint8)

    overlay = input_rgb.copy()
    colors = {1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0)}
    for cls, color in colors.items():
        mask = (mask_display == cls).astype(np.uint8)
        colored_mask = np.stack([mask * c for c in color], axis=-1)
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

    # Save images
    uid = uuid.uuid4().hex
    input_path = f"static/outputs/input_{uid}.png"
    mask_path = f"static/outputs/mask_{uid}.png"
    overlay_path = f"static/outputs/overlay_{uid}.png"

    cv2.imwrite(input_path, (input_display * 255).astype(np.uint8))
    cv2.imwrite(mask_path, (mask_display * 85).astype(np.uint8))
    cv2.imwrite(overlay_path, overlay)

    return {
        'input': input_path,
        'mask': mask_path,
        'overlay': overlay_path
    }
