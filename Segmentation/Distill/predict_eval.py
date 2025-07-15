import os
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
#from models.unet import get_unet
from models.unetpp import get_unetpp  # ✅ 新增
#from models.Attention_unet import AttentionUNet  # ✅ 新增
#from models.Res_unet import ResUNet  # ✅ 新增
import argparse
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# ======= 工具函数 =======

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    img_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    return img_tensor, image_np

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return (mask > 127).astype(np.uint8)

def dice_score(pred, target):
    pred = (pred > 0.5).astype(np.uint8)
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def iou_score(pred, target):
    pred = (pred > 0.5).astype(np.uint8)
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / (union + 1e-8)

def hausdorff_distance(pred_mask, gt_mask):
    pred_pts = np.argwhere(pred_mask)
    gt_pts = np.argwhere(gt_mask)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan  # 避免空掩码错误
    hd_forward = directed_hausdorff(pred_pts, gt_pts)[0]
    hd_backward = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(hd_forward, hd_backward)

def visualize_results(image_np, gt_mask, pred_mask, save_path, filename):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{filename}.png"))
    plt.close()

# ======= 主程序入口 =======

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="test_images", help="Path to test images")
    parser.add_argument("--mask_dir", default="test_masks", help="Path to ground truth masks")
    parser.add_argument("--model_path", default="trained_unet_from_sam.pth", help="Trained UNet model path")
    parser.add_argument("--vis_dir", default="vis_results", help="Directory to save visualizations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = get_unet().to(device)

    model = get_unetpp().to(device)
    #model =AttentionUNet().to(device)
    #model =ResUNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg'))])
    dice_scores = []
    iou_scores = []
    hd_scores = []

    for file in tqdm(image_files, desc="Evaluating"):
        image_path = os.path.join(args.image_dir, file)
        mask_path = os.path.join(args.mask_dir, file)

        if not os.path.exists(mask_path):
            print(f"⚠️ Missing mask for {file}, skipping.")
            continue

        img_tensor, image_np = load_image(image_path)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = pred.squeeze().cpu().numpy()

        gt_mask = load_mask(mask_path)

        dice_scores.append(dice_score(pred_mask, gt_mask))
        iou_scores.append(iou_score(pred_mask, gt_mask))
        hd_scores.append(hausdorff_distance(pred_mask > 0.5, gt_mask))

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_hd = np.nanmean(hd_scores)  # 忽略 nan 值

    print(f"\n📊 评估结果：")
    print(f"平均 Dice 分数: {avg_dice:.4f}")
    print(f"平均 mIoU 分数: {avg_iou:.4f}")
    print(f"平均 Hausdorff 距离: {avg_hd:.4f}")

    # 🎨 可视化 20 个样本
    print(f"\n🎨 保存可视化样本至 {args.vis_dir}")
    sampled_files = random.sample(image_files, min(20, len(image_files)))

    for file in sampled_files:
        image_path = os.path.join(args.image_dir, file)
        mask_path = os.path.join(args.mask_dir, file)

        if not os.path.exists(mask_path):
            continue

        img_tensor, image_np = load_image(image_path)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        gt_mask = load_mask(mask_path)
        visualize_results(image_np, gt_mask, pred_mask, args.vis_dir, os.path.splitext(file)[0])

if __name__ == "__main__":
    main()
