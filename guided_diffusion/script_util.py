import argparse
import torch
import torch.nn.functional as F

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel
from guided_diffusion.restorer.restorer import Restorer
from guided_diffusion.codeformer.codeformer_arch import CodeFormer
from guided_diffusion.face_parsing.bisenet import BiSeNet
from .iresnet import iresnet50

SUPPORTED_TASKS = ['restoration', 'old_photo_restoration', 'old_photo_restoration_pseudo']


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=512,
        num_channels=32,
        num_res_blocks=[1, 2, 2, 2, 2, 3, 4],
        num_heads=1,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32, 16, 8",
        channel_mult="1, 2, 4, 8, 8, 16, 16",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds), # ?
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None, #(NUM_CLASSES if class_cond else None)
        #use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        #num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_restorer():

    return Restorer(
        in_channels=9,
        out_channels=3,
        mid_channels=64,
        num_blocks=23,
        growth_channels=32,
        upscale_factor=1
    )

def create_restorer_code():
    
    return CodeFormer(
        vqgan_path='./models/vqgan/vqgan_code1024.pth'
    )

def create_face_pasring():
    return BiSeNet(num_classes=19, backbone_name='resnet34')


def create_arcface_embedding():
    return iresnet50(False)
    

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def avg_grayscale(img):
    rgb_mean = torch.mean(img, [1], keepdim=True).expand(-1,3,-1,-1)
    return rgb_mean

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    
    # style 0 (default): celebA-HQ avg
    style_mean = torch.tensor([0.03202754, -0.16308397, -0.26475719]).reshape(1,3,1,1).cuda()
    style_std = torch.tensor([0.53549316, 0.47539538, 0.46814889]).reshape(1,3,1,1).cuda()
    # style 1: celebA-HQ 00896
    # style_mean = torch.tensor([0.05546914, -0.22203779, -0.48434922]).reshape(1,3,1,1).cuda()
    # style_std = torch.tensor([0.60425657, 0.53711116, 0.45434347]).reshape(1,3,1,1).cuda()
    # style 2: celebA-HQ 00917
    # style_mean = torch.tensor([-0.43500257, -0.65664947, -0.7156179]).reshape(1,3,1,1).cuda()
    # style_std = torch.tensor([0.47355509, 0.29044732, 0.2357967]).reshape(1,3,1,1).cuda()
    # style 3: celebA-HQ 00439
    # style_mean = torch.tensor([-0.35427222, -0.41545531, -0.36199731]).reshape(1,3,1,1).cuda()
    # style_std = torch.tensor([0.42282647, 0.33328968, 0.36431086]).reshape(1,3,1,1).cuda()

    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)



def adjust_contrast(x, factor=2.5):
    mean = x.mean(dim=(2,3), keepdim=True)
    return (x - mean) * factor + mean


def get_region_mask_3ch(parsing_logits, mask_labels=[0, 14, 16], without=True):
    """
    Returns a 3-channel binary mask:
    - Value 1: regions specified in `mask_labels`
    - Value 0: other regions
    All three channels are the same.
    0: 背景  14: 脖子  16: 衣服

    Args:
        parsing_logits: torch.Tensor of shape [B, C, H, W], raw output from segmentation model
        mask_labels: list of int, label indices to be treated as masked regions
    """
    save_parsing_labelmap_for_eval(parsing_logits)
    visualize_parsing_labels_to_file(parsing_logits)
    label_map = parsing_logits.argmax(dim=1)  # [B, H, W]
    mask_single = torch.zeros_like(label_map, dtype=torch.bool)
    
    for label in mask_labels:
        mask_single |= (label_map == label)

    if not without:
        mask_single = ~mask_single

    region_mask = mask_single.float() .unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
    return region_mask


import numpy as np
def combine_with_damage_mask(region_mask_3ch, damage_mask_img):
    """
    Merges the region mask with the mask of the breakage detection.
    Parameters.
        region_mask_3ch: [B, 3, H, W]
        damage_mask_img: np.ndarray [H, W, 3], from cv2.imread(...) / 255., consistent across all three channels
    Returns.
        [B, 3, H, W], values 0 or 1
    """
    assert isinstance(damage_mask_img, np.ndarray) and damage_mask_img.ndim == 3 and damage_mask_img.shape[2] == 3
    B, C, H, W = region_mask_3ch.shape

    # Turn tensor, shape [1, 3, H, W]
    damage_tensor = torch.from_numpy(damage_mask_img).permute(2, 0, 1).unsqueeze(0).to(region_mask_3ch.device)
    damage_tensor = (damage_tensor > 0.5).float()

    # Since the original image is grayscale expanded to RGB, the three channels are the same → we only take channel 0 as the unity mask
    binary_mask = damage_tensor[:, 0:1, :, :]  # [1, 1, H, W]
    binary_mask = binary_mask.expand(B, 3, H, W)  # expand to [B, 3, H, W]

    fused_mask = torch.clamp(region_mask_3ch + binary_mask, 0, 1)
    return fused_mask


def extract_background_damage_3ch(bg_mask, damage_mask):
    """
    提取背景区域上的破损掩码，并复制为3通道。

    Args:
        bg_mask (Tensor): 背景掩码，[B, 3, H, W]，背景为0，其它为1
        damage_mask (Tensor): 破损掩码，[B, 3, H, W]，破损为1，其它为0

    Returns:
        Tensor: 新掩码，仅在背景区域且破损的位置为1，其它为0，形状为 [B, 3, H, W]
    """
    # 将背景掩码和破损掩码转换为单通道
    bg_mask_single = bg_mask[:, 0:1, :, :]  # [B, 1, H, W]
    damage_mask_single = damage_mask[:, 0:1, :, :]  # [B, 1, H, W]

    # 仅保留背景区域上的破损
    background_region = (bg_mask_single == 0)
    new_mask = damage_mask_single * background_region.float()  # [B, 1, H, W]

    # 扩展为 3 通道
    new_mask_3ch = new_mask.repeat(1, 3, 1, 1)  # [B, 3, H, W]
    return new_mask_3ch

def expand_mask_vertically(mask, expand_y=8, expand_x=8):
    """
    在垂直（y轴）和水平（x轴）方向上扩展破损区域。

    Args:
        mask (Tensor): 输入掩码，形状为 [B, C, H, W]，值为0或1
        expand_y (int): 上下扩展的像素数
        expand_x (int): 左右扩展的像素数

    Returns:
        Tensor: 扩展后的掩码，形状与输入相同
    """
    B, C, H, W = mask.shape
    mask_expanded = mask.clone()

    # 垂直方向扩展（上下）
    for i in range(1, expand_y + 1):
        up = F.pad(mask[:, :, :-i, :], (0, 0, i, 0), value=0)
        down = F.pad(mask[:, :, i:, :], (0, 0, 0, i), value=0)
        mask_expanded = torch.logical_or(mask_expanded, up)
        mask_expanded = torch.logical_or(mask_expanded, down)

    # 水平方向扩展（左右）
    for j in range(1, expand_x + 1):
        left = F.pad(mask[:, :, :, :-j], (j, 0, 0, 0), value=0)
        right = F.pad(mask[:, :, :, j:], (0, j, 0, 0), value=0)
        mask_expanded = torch.logical_or(mask_expanded, left)
        mask_expanded = torch.logical_or(mask_expanded, right)

    return mask_expanded.float()



import torch.nn.functional as F
def compute_pixel_diff(img, mask):
    """
    Compute per-pixel gradient magnitude (horizontal + vertical)
    only inside the mask (mask == 1), and ensure no gradient leak outside.

    Args:
        img  (Tensor): (B, C, H, W) input image.
        mask (Tensor): (B, 1, H, W) binary mask (float or bool), 1 = valid region.

    Returns:
        Tensor: (B, 1, H, W) pixel-wise diff map (average across channels).
    """
    # Ensure float mask
    mask = mask.float()

    # Cut gradient outside mask
    img = img * mask + img.detach() * (1 - mask)

    # Vertical and horizontal diffs
    diff_v = (img[:, :, :-1, :] - img[:, :, 1:, :])
    diff_h = (img[:, :, :, :-1] - img[:, :, :, 1:])

    # Valid area: only where both neighboring pixels are inside the mask
    valid_v = mask[:, :, :-1, :] * mask[:, :, 1:, :]
    valid_h = mask[:, :, :, :-1] * mask[:, :, :, 1:]

    # Multiply mask and take absolute value
    diff_v = diff_v.abs() * valid_v
    diff_h = diff_h.abs() * valid_h

    # Average over channels (C) to get grayscale-like gradient map
    diff_v = diff_v.mean(1, keepdim=True)
    diff_h = diff_h.mean(1, keepdim=True)

    # Pad back to original shape
    diff_v = F.pad(diff_v, (0, 0, 0, 1))  # pad bottom row
    diff_h = F.pad(diff_h, (0, 1, 0, 0))  # pad right column

    # Combine
    return diff_v + diff_h  # (B, 1, H, W)


def tv_loss(x, mask):
    """
    Compute total variation loss within the masked region only.

    Args:
        x (Tensor): (B, C, H, W)
        mask (Tensor): (B, 1, H, W)

    Returns:
        Scalar tensor (float): TV loss in masked region.
    """
    mask = mask.float()
    diff_h = ((x[:, :, :, :-1] - x[:, :, :, 1:]) ** 2) * mask[:, :, :, :-1] * mask[:, :, :, 1:]
    diff_v = ((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2) * mask[:, :, :-1, :] * mask[:, :, 1:, :]

    return (diff_h.sum() + diff_v.sum()) / (mask.sum() + 1e-6)  # normalize to avoid scale explosion




import matplotlib
matplotlib.use("Agg") 
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image


def save_parsing_labelmap_for_eval(parsing_logits, save_path="parsing_label_eval.png"):
    """
    Save raw label map from parsing logits as a grayscale PNG (each pixel = class ID),
    suitable for IoU / Dice evaluation.
    
    Args:
        parsing_logits: [1, C, H, W] torch tensor
        save_path: output path for the label map
    """
    label_map = parsing_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # [H, W]
    label_image = Image.fromarray(label_map, mode='L')  # 'L' mode = grayscale
    label_image.save(save_path)

def visualize_parsing_labels_to_file(parsing_logits, save_path="parsing_labels.png", show_labels=None, figsize=(8, 8)):
    """
    save parsing label map
    """
    label_map = parsing_logits.argmax(dim=1)[0].cpu().numpy()  # [H, W]

    H, W = label_map.shape
    label_ids = np.unique(label_map)

    if show_labels is not None:
        label_ids = [l for l in label_ids if l in show_labels]

    colormap = plt.get_cmap("tab20", 20)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 20), ncolors=20)

    plt.figure(figsize=figsize)
    plt.imshow(label_map, cmap=colormap, norm=norm)
    # cbar = plt.colorbar(ticks=label_ids)
    # # cbar.set_label("Label ID")
    # cbar.set_ticks(label_ids)
    # cbar.set_ticklabels([f"{i}" for i in label_ids], fontsize=32)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # # 1. 解析 label map
    # label_map = parsing_logits.argmax(dim=1)[0].cpu().numpy()  # [H, W]

    # # 2. 使用 matplotlib 的 colormap 映射为 RGB 图像（值归一化）
    # import matplotlib.pyplot as plt
    # cmap = plt.get_cmap("tab20", 20)
    # label_color = (cmap(label_map / 19.0)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3] RGB

    # # 3. RGB → BGR for OpenCV
    # label_bgr = cv2.cvtColor(label_color, cv2.COLOR_RGB2BGR)

    # # 4. 无边保存
    # cv2.imwrite(save_path, label_bgr)
    # print(f"✅ Saved parsing label (no white border): {save_path}")


def gram_matrix(tensor):
    """Calculating the Gram Matrix"""
    B, C, H, W = tensor.size()
    tensor = tensor.view(B, C, -1)
    gram = torch.bmm(tensor, tensor.transpose(1, 2))
    gram = gram / (C * H * W) 
    return gram

def color_style_loss(pred_img, target_img, vgg_extractor, layer_idx=0):
    """VGG-based color style migration loss"""
    pred_features = vgg_extractor(pred_img)
    target_features = vgg_extractor(target_img)

    pred_feat = pred_features[layer_idx]
    target_feat = target_features[layer_idx]

    loss = F.mse_loss(gram_matrix(pred_feat), gram_matrix(target_feat), reduction='sum') * 5000000
    return loss
    


def adaptive_instance_normalization1(content_feat, style_feat):
    
    # style 0 (default): celebA-HQ avg
    style_mean, style_std = calc_mean_std(style_feat)
    # style_mean = torch.tensor([0.03202754, -0.16308397, -0.26475719]).reshape(1,3,1,1).cuda()
    # style_std = torch.tensor([0.53549316, 0.47539538, 0.46814889]).reshape(1,3,1,1).cuda()

    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    
    return normalized_feat * 1 * style_std.expand(size) + style_mean.expand(size)






def random_flip_mask(mask, ratio=0.1):
    """
    Randomly flip a given ratio of white pixels (1->0) in a 3-channel binary mask.
    Operates consistently across all 3 channels.
    mask: Tensor [B, 3, H, W] with values 0 or 1
    """
    # Work on a single channel
    mask_1ch = mask[:, 0:1, :, :].clone()

    # Identify white positions
    white_positions = mask_1ch == 1

    # Generate random values
    rand = torch.rand_like(mask_1ch.float())

    # Random flip on white positions
    flip_mask = (rand < ratio) & white_positions

    # Set flipped positions to 0
    mask_1ch[flip_mask] = 0

    # Repeat to 3 channels
    return mask_1ch.repeat(1, 3, 1, 1)