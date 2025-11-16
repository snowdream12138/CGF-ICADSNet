import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io, transform
import torch
import torch.nn.functional as F
from tqdm import tqdm

from segment_anything import sam_model_registry
import argparse
from sam_lora_image_encoder import LoRA_Sam
from typing import List


# visualization functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=1)
    )

@torch.no_grad()
def medsam_inference(medsam_model, image_features_list: List[torch.Tensor], box_1024, H, W):
    device = image_features_list[0].device
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.sam.prompt_encoder(
        points=None, boxes=box_torch, masks=None
    )

    low_res_logits, _ = medsam_model.sam.mask_decoder(
        image_embeddings=image_features_list,
        image_pe=medsam_model.sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def load_bbox_coordinates(file_path):
    """
    Load bounding box coordinates from a file.

    Args:
        file_path (str): Path to the file containing bounding box coordinates.

    Returns:
        dict: A dictionary mapping image filenames to bounding box coordinates.
    """
    bbox_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line into filename and coordinates
            parts = line.strip().split(':')
            if len(parts) != 2:
                print(f"Warning: Invalid line format: {line}")
                continue

            image_name = parts[0].strip()
            coords_str = parts[1].strip()

            # Remove any extra characters (e.g., spaces, brackets)
            coords_str = coords_str.replace('[', '').replace(']', '').replace(' ', '')

            # Split the coordinates into individual values
            try:
                coords = [int(coord) for coord in coords_str.split(',')]
            except ValueError as e:
                print(f"Warning: Invalid coordinates in line: {line}. Error: {e}")
                continue

            # Ensure there are exactly 4 coordinates (x1, y1, x2, y2)
            if len(coords) != 4:
                print(f"Warning: Invalid number of coordinates in line: {line}")
                continue

            # Store the coordinates as a single box
            bbox_dict[image_name] = [coords]

    return bbox_dict

def process_images(image_folder, bbox_file, output_folder, medsam_model, device):
    bbox_dict = load_bbox_coordinates(bbox_file)

    for image_name, boxes in tqdm(bbox_dict.items()):
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist, skipping...")
            continue

        # Load image
        img_np = io.imread(image_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape

        # Preprocess image
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        # Get image embedding
        with torch.no_grad():
            image_embedding_list = medsam_model.sam.image_encoder(img_1024_tensor)

        # Process each box
        for i, box_np in enumerate(boxes):
            box_1024 = np.array(box_np) / np.array([W, H, W, H]) * 1024

            medsam_seg = medsam_inference(medsam_model, image_embedding_list, [box_1024], H, W)

            # Save segmentation
            seg_output_path = os.path.join(
                output_folder, f"seg_{os.path.splitext(image_name)[0]}_{i}.png"
            )
            io.imsave(seg_output_path, medsam_seg * 255, check_contrast=False)
            print(f"Saved segmentation to {seg_output_path}")


# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for MedSAM")
    parser.add_argument(
        "-i", "--image_folder", type=str, required=True, help="Path to the image folder"
    )
    parser.add_argument(
        "-b", "--bbox_file", type=str, required=True, help="Path to the bbox file"
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, required=True, help="Path to the output folder"
    )
    parser.add_argument(
        "-chk",
        "--checkpoint",
        type=str,
        default="/root/MedSAM-main/work_dir/unet+AG+CBAM+SE+laplacian_loss_new/medsam_lora_best.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    parser.add_argument(
        "--base_model",
        type=str,
        default="/root/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to official SAM base model checkpoint"
    )

    args = parser.parse_args()

    base_sam_checkpoint = "/root/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth"
    sam = sam_model_registry["vit_b"](checkpoint=args.base_model)
    medsam_model = LoRA_Sam(sam, r=16)

    medsam_model.load_lora_parameters(args.checkpoint)

    medsam_model = medsam_model.to(args.device)
    medsam_model.eval()


    os.makedirs(args.output_folder, exist_ok=True)

    # Process images
    process_images(
        args.image_folder, args.bbox_file, args.output_folder, medsam_model, args.device
    )


