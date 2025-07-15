import torch



torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import glob
import argparse
import cv2
from PIL import Image
from ultralytics import YOLO

from models.unetpp import get_unetpp  # ✅ 新增

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ========== 设置设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 图像预处理函数 ==========

def preprocess_mri(image_np):
    if image_np.ndim == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    image_norm = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX)
    image_rgb = cv2.cvtColor(image_norm.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return image_rgb


    
# ========== 加载模型 ==========

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam)


unet = get_unetpp().to(device)

optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_add", default="./detect/p/weights/best.pt")
args = parser.parse_args()
yolo_model = YOLO(args.checkpoint_add)

# ========== Hook ==========
teacher_features = []
def teacher_hook_fn(module, input, output):
    teacher_features.append(output)
hook_teacher = sam.image_encoder.trunk.blocks[5].register_forward_hook(teacher_hook_fn)

student_features = []
def student_hook_fn(module, input, output):
    student_features.append(output)
hook_student = unet.finalconv.register_forward_hook(student_hook_fn)

# ========== 加载图像列表 ==========
image_list = glob.glob("./p/1/train/images/*.jpg")
print(f"共加载训练图像: {len(image_list)} 张")

# ========== 损失函数 ==========
bce_loss_fn = nn.BCEWithLogitsLoss()
mse_loss_fn = nn.MSELoss()
beta = 0.1

# ========== ULWR 损失函数 ==========
def compute_weights(variance, epoch):
    eps = 1e-6
    w1 = 1.0 / (variance + eps)
    w2 = torch.sigmoid(1 - variance)
    w3 = 0.4     
    w_sum = w1 + w2 + w3
    w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
    return w1, w2, w3

# ========== 训练循环 ==========
import scipy.ndimage
for epoch in range(100):
    for i, img_path in enumerate(image_list):
        image = Image.open(img_path).convert("RGB")
        image_np_raw = np.array(image)
        image_np = preprocess_mri(image_np_raw)

        results = yolo_model.predict([image], imgsz=256, conf=0.5)
        boxes = results[0].boxes
        if boxes.shape[0] == 0:
            continue

        predictor.set_image(image_np)
        combined_soft_mask = np.zeros(image_np.shape[:2], dtype=np.float32)
        sigma2_all = []
        for box in boxes.xyxy.cpu().numpy():
            masks, iou_preds, low_res_logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
                return_logits=True
            )
            
                   
            prob_mask = torch.sigmoid(torch.from_numpy(low_res_logits[0])).numpy()
            sigma2 = np.var(prob_mask)
            sigma2_all.append(sigma2)
            prob_mask_resized = cv2.resize(prob_mask, (image_np.shape[1], image_np.shape[0]))
            combined_soft_mask = np.maximum(combined_soft_mask, prob_mask_resized)

        sigma2_batch = np.mean(sigma2_all)
        sigma2_tensor = torch.tensor(sigma2_batch).to(device)

        img_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        soft_mask_tensor = torch.from_numpy(combined_soft_mask).unsqueeze(0).unsqueeze(0).float()
        img_tensor, soft_mask_tensor = img_tensor.to(device), soft_mask_tensor.to(device)

        pred_logits = unet(img_tensor)
        pred_probs = torch.sigmoid(pred_logits)

        bce_loss = bce_loss_fn(pred_logits, soft_mask_tensor)
        prob_distill_loss = mse_loss_fn(pred_probs, soft_mask_tensor)

        teacher_feat = teacher_features.pop()
        student_feat = student_features.pop()
        if teacher_feat.shape != student_feat.shape:
            teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[-2:], mode='bilinear')
        feat_distill_loss = F.mse_loss(student_feat, teacher_feat.detach())

        w1, w2, w3 = compute_weights(sigma2_tensor, epoch)
        avg_w = (w1 + w2 + w3) / 3.0
        reg = beta * ((w1 - avg_w) ** 2 + (w2 - avg_w) ** 2 + (w3 - avg_w) ** 2)
        total_loss = w1 * bce_loss + w2 * prob_distill_loss + w3 * feat_distill_loss
        final_loss = total_loss + reg
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch}] Img {i} | Loss: {final_loss.item():.4f} (BCE: {bce_loss.item():.4f}, Distill: {prob_distill_loss.item():.4f}, Feat: {feat_distill_loss.item():.4f}) | w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}")

   
torch.save(unet.state_dict(), "sam2-unetpp-3-6.pth")
print("模型保存完毕：sam2-unetpp-3-6.pth")
# 清理 hook
hook_teacher.remove()
hook_student.remove()
