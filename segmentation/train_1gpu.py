# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from sam_lora_image_encoder import LoRA_Sam
from adan_pytorch import Adan
import cv2
from monai.losses import DiceLoss, SSIMLoss

torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


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
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

def parse_layer_range(layer_str):
    if layer_str == 'all':
        return None
    layers = []
    for part in layer_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            layers.extend(range(start, end+1))
        else:
            layers.append(int(part))
    return sorted(set(layers))

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


# %% sanity test of dataset class
tr_dataset = NpyDataset("/media/data2/yuanjianye/MedSAM-main/data/Yolo_dataset")
tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True)
for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
    print(image.shape, gt.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, 3)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    # set title
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 3)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    # set title
    axs[1].set_title(names_temp[idx])
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    # plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="/media/data2/yuanjianye/MedSAM-main/data/Yolo_dataset",
    help="path to training npy files; two subfolders: gts and imgs",
)
# parser.add_argument('-thd', type=bool, default=False, help='3d or not')
# parser.add_argument('-mid_dim', type=int, default=None, help='middle dim of adapter or the rank of lora matrix')
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B_new")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-checkpoint", type=str, default="/media/data2/yuanjianye/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth")
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument("--load_pretrain", type=bool, default=True, help="load pretrain model")
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="/media/data2/yuanjianye/MedSAM-main/work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("-use_wandb", type=bool, default=False, help="use wandb to monitor training")
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--lora_rank", type=int, default=16, help="LoRA size")
parser.add_argument("--lora_ckpt", type=str, default=None, help="Lora path")

parser.add_argument("--lora_layer", type=str, default='all', help="such as '0-4,8-11'")
args = parser.parse_args()


if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model


class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder, lora_model=None):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        self.lora_model = lora_model

    def forward(self, image, box):

        image_embedding_list = self.image_encoder(image)

        box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        bbox_prompt_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

        sparse_embeddings, dense_embeddings = bbox_prompt_embeddings

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding_list,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, pred):
        diff_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        diff_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        tv_loss = torch.mean(diff_x) + torch.mean(diff_y)
        return tv_loss

class LaplacianLoss(nn.Module):
    def __init__(self, max_levels=3, k_size=5, sigma=2.0):
        super(LaplacianLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = self.build_gauss_kernel(size=self.k_size, sigma=self.sigma)

    def build_gauss_kernel(self, size=5, sigma=1.0, n_channels=1):
        if size % 2 != 1:
            raise ValueError("kernel size must be an odd number.")
        kernel = cv2.getGaussianKernel(size, sigma)
        kernel = np.dot(kernel, kernel.T)
        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()
        kernel = kernel.repeat(n_channels, 1, 1, 1)
        return kernel

    def laplacian_pyramid(self, img, kernel, max_levels):
        current = img
        pyramid = []
        for _ in range(max_levels):
            kernel = kernel.to(current.device)
            filtered = F.conv2d(current, kernel, stride=1, padding='same', groups=current.shape[1])
            lap = current - filtered
            pyramid.append(lap)
            current = F.avg_pool2d(filtered, kernel_size=2, stride=2)
        pyramid.append(current)
        return pyramid

    def forward(self, pred, gt):
        pred_pyramid = self.laplacian_pyramid(pred, self._gauss_kernel, self.max_levels)
        gt_pyramid = self.laplacian_pyramid(gt, self._gauss_kernel, self.max_levels)

        loss = 0
        for i in range(len(pred_pyramid)):
            loss += F.l1_loss(pred_pyramid[i], gt_pyramid[i]) * (2 ** i)

        return loss / len(pred_pyramid)

def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)

    # 创建保存路径
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    lora_model = LoRA_Sam(
        sam_model,
        r=args.lora_rank,
        lora_layer=parse_layer_range(args.lora_layer)
    ).to(device)

    medsam_model = MedSAM(
        image_encoder=lora_model.sam.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        lora_model=lora_model,
    ).to(device)

    if args.lora_ckpt:
        lora_model.load_lora_parameters(args.lora_ckpt)


    medsam_model.train()
    for name, param in medsam_model.mask_decoder.named_parameters():
        param.requires_grad = True
    trainable_params = list(filter(lambda p: p.requires_grad, medsam_model.parameters()))


    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252


    optimizer = Adan(
        # img_mask_encdec_params,
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print("Trainable Parameters in Image Encoder:")
    for name, param in medsam_model.image_encoder.named_parameters():
        if param.requires_grad:
            print(f"\t{name}")

    print("Trainable Parameters in Prompt Decoder:")
    for name, param in medsam_model.prompt_encoder.named_parameters():
        if param.requires_grad:
            print(f"\t{name}")

    print("Trainable Parameters in Mask Decoder:")
    for name, param in medsam_model.mask_decoder.named_parameters():
        if param.requires_grad:
            print(f"\t{name}")



    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")


    tv_loss = TVLoss()
    tv_loss_weight = 0.5


    laplacian_loss = LaplacianLoss()
    laplacian_loss_weight = 0.5

    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10


    train_dataset = NpyDataset(args.tr_npy_path)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)


                    medsam_pred_prob = torch.sigmoid(medsam_pred)
                    loss_laplacian = laplacian_loss(medsam_pred_prob, gt2D.float())


                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float()) + tv_loss_weight * tv_loss(medsam_pred) + laplacian_loss_weight * loss_laplacian

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                medsam_pred = medsam_model(image, boxes_np)
                medsam_pred_prob = torch.sigmoid(medsam_pred)
                loss_laplacian = laplacian_loss(medsam_pred_prob, gt2D.float())

                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float()) + tv_loss_weight * tv_loss(medsam_pred) + laplacian_loss_weight * loss_laplacian
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        if epoch % 20 == 0:

            save_mode_path = os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth')
            medsam_model.lora_model.save_lora_parameters(save_mode_path)
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss

            save_mode_path = os.path.join(model_save_path, 'medsam_lora_best.pth')
            medsam_model.lora_model.save_lora_parameters(save_mode_path)

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
