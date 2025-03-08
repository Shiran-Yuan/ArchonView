import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import math
from tqdm import tqdm
from torchvision.transforms import transforms
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
MODEL_DEPTH = 24
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', 'path/to/checkpoint.pth'
patch_nums = (1,2,3,4,5,6,8,10,13,16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        depth=MODEL_DEPTH, shared_aln=False,
    )
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu')['trainer']['var_wo_ddp'], strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')
import clip
device = "cuda"
model, preprocess = clip.load("ViT-L/14", device=device)

seed = 42
cfg = 3

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')
aug = [transforms.ToTensor(), lambda x: x.add(x).add_(-1)]
B = 1

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2])
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])
def get_T(target_RT, cond_RT):
    R1, T1 = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R1.T @ T1
    R2, T2= cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R2.T @ T2
    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T.unsqueeze(0)

if __name__ == '__main__':
    img = PImage.open(f'path/to/input_image.png')
    img = img.convert('RGB')
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        emb = image_features / image_features.norm(dim=-1, keepdim=True)
    transform = transforms.Compose(aug)
    source = transform(img).unsqueeze(0).to(device)
    src_idx_Bl = vae.img_to_idxBl(source)
    src_BL = torch.cat(src_idx_Bl, dim=1)
    src_BLCv = vae.quantize.embedding(src_BL)
    cond = np.load('path/to/input_pose.npy')
    target = np.load('path/to/target_pose.npy')
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            recon_B3HW = var.autoregressive_infer_cfg(B=B, emb=emb, y_BLCv=src_BLCv, pose=get_T(target, cond).to(device), cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=False)
    recon_img = torchvision.transforms.ToPILImage()(recon_B3HW.squeeze(0).cpu())
    recon_img.save('path/to/target_file.png')
