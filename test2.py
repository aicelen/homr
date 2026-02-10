import cv2
import numpy as np
import onnxruntime as ort
from time import perf_counter

def test_onnx_segmentation(model_path: str, image_path: str, out_path: str = "segmentation_result.png"):
    TILE_SIZE = 320
    NUM_CLASSES = 6  # keep exactly 6 output colors (classes 0..5)

    # BGR palette (change to taste). Must have NUM_CLASSES entries.
    PALETTE = [
        (0, 0, 0),         # class 0 -> black
        (0, 0, 255),       # class 1 -> red
        (0, 255, 0),       # class 2 -> green
        (255, 0, 0),       # class 3 -> blue
        (0, 255, 255),     # class 4 -> yellow (BGR)
        (255, 0, 255),     # class 5 -> magenta
    ]
    assert len(PALETTE) == NUM_CLASSES, "Palette length must equal NUM_CLASSES"

    # -----------------------------
    # Load model
    # -----------------------------
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # -----------------------------
    # Load image as GRAYSCALE first (like working code)
    # -----------------------------
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert to BGR (working code expectation)
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    h, w, _ = image_bgr.shape

    # -----------------------------
    # Prepare output mask
    # -----------------------------
    full_mask = np.zeros((h, w), dtype=np.uint8)

    # -----------------------------
    # Tile inference
    # -----------------------------
    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            tile = image_bgr[y:y + TILE_SIZE, x:x + TILE_SIZE]

            # Pad tile if needed WITH WHITE (255), not black!
            pad_h = TILE_SIZE - tile.shape[0]
            pad_w = TILE_SIZE - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )

            # Preprocess (HWC -> CHW, float32, NO NORMALIZATION!)
            tile_input = tile.astype(np.float32)  # Keep values in [0, 255]
            tile_input = np.transpose(tile_input, (2, 0, 1))
            tile_input = np.expand_dims(tile_input, axis=0)

            # Inference
            t0 = perf_counter()
            output = session.run([output_name], {input_name: tile_input})[0]
            print(f"Inference time: {perf_counter() - t0:.4f}s")

            # -----------------------------
            # Postprocess
            # -----------------------------
            # Multi-class segmentation (argmax across class dimension)
            pred = np.argmax(output[0], axis=0).astype(np.uint8)

            # Remove padding: pred shape is TILE_SIZE x TILE_SIZE, so crop back to original tile size
            crop_h = TILE_SIZE - pad_h if pad_h > 0 else TILE_SIZE
            crop_w = TILE_SIZE - pad_w if pad_w > 0 else TILE_SIZE
            pred = pred[:crop_h, :crop_w]

            full_mask[y:y + pred.shape[0], x:x + pred.shape[1]] = pred

    # -----------------------------
    # Ensure mask contains only classes 0..NUM_CLASSES-1
    # -----------------------------
    unique_vals = np.unique(full_mask)
    if unique_vals.size == 0:
        print("Warning: empty segmentation mask.")
    if unique_vals.max() >= NUM_CLASSES:
        print(f"Warning: found class ids {unique_vals[unique_vals >= NUM_CLASSES]} >= {NUM_CLASSES}; clipping to range 0..{NUM_CLASSES-1}.")
    # clip any unexpected labels to the valid range
    mask_clipped = np.clip(full_mask, 0, NUM_CLASSES - 1).astype(np.uint8)

    # -----------------------------
    # Build colored image using the palette
    # -----------------------------
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(NUM_CLASSES):
        color = PALETTE[cls_idx]  # BGR
        color_mask[mask_clipped == cls_idx] = color

    # Save using OpenCV (BGR expected): color_mask is already BGR
    saved = cv2.imwrite(out_path, color_mask)
    if not saved:
        raise IOError(f"Failed to write output image to {out_path}")

    print(f"Saved colored segmentation to: {out_path}")
    print(f"Unique mask values (original): {unique_vals}")
    print(f"Unique mask values (after clipping to 0..{NUM_CLASSES-1}): {np.unique(mask_clipped)}")

if __name__ == "__main__":
    # segnet_155-1240eedca553155b3c75fc9c7f643465383430a0
    # segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f
    test_onnx_segmentation(
        "/home/enno/Documents/GitHub/homr/homr/segmentation/segnet_155-1240eedca553155b3c75fc9c7f643465383430a0.onnx",
        "OdeToJoy.jpg",
        out_path="segmentation_result.png"
    )
