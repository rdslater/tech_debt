import os
import torch
import pydicom
import numpy as np
import pandas as pd
from torchvision import transforms
import cv2
from ultralytics import YOLO
from PIL import Image
import traceback
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import StrongModel, SwinBinaryClassifier


def get_transform():
    """Returns Albumentations transform for flipping and rotating."""
    return A.Compose([
        ToTensorV2()  # Converts to PyTorch tensor
    ])


def hflip(tensor):
    tensor = tensor.flip(2)
    return tensor


def dicom_to_pil(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    if img.size == 0:
        raise ValueError("Empty pixel array")

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(img)


swin_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def main():
    segmentation_model_path = "50epochs_strongmodel_augmentation_gencropping_noclahe_8020split_mitUNET.pth"
    segmentation_model = StrongModel('Unet',
                                     'mit_b4',
                                     in_channels=3,
                                     out_classes=1,
                                     encoder_weights=None)

    # Load state dictionary from checkpoint
    checkpoint = torch.load(segmentation_model_path,  map_location="cpu")
    segmentation_model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    swin_model_path = "swintiny_binary_best.pth"
    yolo_model_path = "Yolo1.pt"
    output_dir = "./screenshotscropped"
    img_size = 512  # this is for YOLO, it should be 512
    crop_size = 512  # this is the size after cropping
    EXCEL_PATH = r"./data.xlsx"
    df = pd.read_excel(EXCEL_PATH)
    crop_output_dir = os.path.join(output_dir, "cropped")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(crop_output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = {0: "Left", 1: "Right"}
    #  Yolo will start with a confidence of 0.5, then if it doesn't detect anything, it will go down with steps of 0.1, then if it reaches 0.1 without detection
    #  then it will go through the values in the list below
    start_conf = 0.5
    step = 0.1
    special_after_0_1 = [0.1, 0.08, 0.05, 0.03, 0.01, 0.005, 0.001]

    # Load Swin eye classification
    swin_model = SwinBinaryClassifier(pretrained=False).to(device)
    checkpoint = torch.load(swin_model_path, map_location=device)
    swin_model.load_state_dict(checkpoint['model_state_dict'])
    swin_model.eval()

    # Init YOLO
    yolo_model = YOLO(yolo_model_path)

    # design the pipeline, first classify the eye, then customize the cropping
    for idx, dicom_path in enumerate(df["File Path"], start=1):
        dicom_path = str(dicom_path).strip()
        base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        print(f"\n[{idx}/{len(df)}] Processing {base_name}")
        if not os.path.exists(dicom_path):
            print("File missing")
            continue
        try:
            img = dicom_to_pil(dicom_path)
        except Exception as e:
            print(f"DICOM read failed: {e}")
            continue
        # 1)is it left or right eye?
        input_tensor = swin_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = swin_model(input_tensor)
            prob = torch.sigmoid(output).item()
            pred_class = int(prob > 0.5)
            eye_label = label_map[pred_class]
        print(f"\n{base_name}: {eye_label} Eye (prob={prob:.3f})")
        # 2)detect the disc
        found_results = None
        conf = start_conf

        while conf > 0.1:
            try:
                results = yolo_model.predict(source=img,
                                             imgsz=img_size,
                                             device=device,
                                             conf=conf,
                                             verbose=False)
            except Exception as e:
                print(f"Predict error at conf={conf} for {base_name}: {e}")
                traceback.print_exc()
                break

            boxes = getattr(results[0], "boxes", None)
            num = len(boxes) if boxes is not None else 0
            if num > 0:
                found_results = (results, conf)
                break
            conf = round(conf - step, 6)

        if found_results is None:
            for conf in special_after_0_1:
                try:
                    results = yolo_model.predict(source=img,
                                                 imgsz=img_size,
                                                 device=device,
                                                 conf=conf,
                                                 verbose=False)
                except Exception as e:
                    print(f"Predict error at conf={conf} for {base_name}: {e}")
                    traceback.print_exc()
                    continue
                boxes = getattr(results[0], "boxes", None)
                num = len(boxes) if boxes is not None else 0
                if num > 0:
                    found_results = (results, conf)
                    break

        if found_results is None:
            results = None
            print(f"No YOLO detections for {base_name}")
            continue

        results, final_conf = found_results
        print(f"YOLO detected {len(results[0].boxes)} boxes at conf={final_conf} for {base_name}")

        # 3)save the cropped version
        save_path = os.path.join(output_dir, f"{base_name}_pred.jpg")
        try:
            results[0].save(filename=save_path)

            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()

            if len(boxes) == 0:
                print(f"No boxes to crop for {base_name}")
                continue

            # keep the detection with highest confidence:
            # This is because each eye will only have 1 disc
            best_idx = np.argmax(scores)
            x1, y1, x2, y2 = boxes[best_idx]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            img_np = np.array(img)
            h, w, _ = img_np.shape

            # Offset by xpixels  left or right of the disc center
            if eye_label == "Left":
                cx_offset = cx + 50  # -30
            else:  # Right eye
                cx_offset = cx - 50  # +30

            # Calculate crop bounds
            x1_crop = max(0, cx_offset - crop_size // 2)
            y1_crop = max(0, cy - crop_size // 2)
            x2_crop = x1_crop + crop_size
            y2_crop = y1_crop + crop_size

            # in case it's out of bound
            if x2_crop > w:
                x1_crop = w - crop_size
                x2_crop = w
            if y2_crop > h:
                y1_crop = h - crop_size
                y2_crop = h

            cropped = img_np[int(y1_crop):int(y2_crop),
                             int(x1_crop):int(x2_crop)]
            cropped_img = Image.fromarray(cropped)
            from torchvision import transforms
            to_tensornew = transforms.ToTensor()
            input_segmentation = to_tensornew(cropped_img).unsqueeze(0).to(device)
            with torch.no_grad():
                mask_logit = segmentation_model(input_segmentation)
                mask_probability = torch.sigmoid(mask_logit)
                mask_prediction = (mask_probability > 0.5).cpu().numpy()[0, 0]
                # print(mask_prediction.shape)
            mask_overlay = np.zeros((512, 512))
            mask_overlay[np.array(mask_prediction) > 0] = 255
            mask_save_path = os.path.join(crop_output_dir, f"{base_name}_{eye_label}_mask.png")
            mask_overlay = ((mask_overlay > 0) * 255).astype(np.uint8)
            img_np = np.array(img)                 # RGB uint8
            img_cv = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            colored_mask = np.zeros_like(img_cv, dtype=np.uint8)
            print(img_cv.shape)
            print(mask_overlay.shape)
            colored_mask[mask_overlay > 0] = (0, 0, 255)
            to_ret = cv2.addWeighted(img_cv, 1 - 0.3, colored_mask, 0.3, 0)
            # Save overlay image
            cv2.imwrite(mask_save_path, to_ret)

        except Exception as e:
            print(f"Failed to process {base_name}: {e}")
            traceback.print_exc()

        print("\nAll predictions + 512×512 directional crops completed.")
    return 0


if __name__ == "__main__":
    main()
