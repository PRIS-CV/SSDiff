import torch
from CAP_VSTNet.models.RevResNet import RevResNet
from CAP_VSTNet.models.cWCT import cWCT
import cv2

def run_style_transfer_tensor(
    content_tensor: torch.Tensor,
    style_tensor: torch.Tensor,
    ckpt_path: str = 'CAP_VSTNet/checkpoints/photo_image.pt',
    mode: str = 'photorealistic',
    content_seg: torch.Tensor = None,
    style_seg: torch.Tensor = None,
    alpha_c: float = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> torch.Tensor:
    """
    Perform style transfer given content/style tensors.
    Args:
        content_tensor: Tensor of shape (1, 3, H, W)
        style_tensor: Tensor of shape (1, 3, H, W)
        ckpt_path: Path to checkpoint
        mode: 'photorealistic' or 'artistic'
        content_seg: Optional segmentation map for content (1, H, W)
        style_seg: Optional segmentation map for style (1, H, W)
        alpha_c: Optional interpolation factor between content and style
        device: torch.device
    Returns:
        stylized_tensor: Tensor of shape (1, 3, H, W)
    """

    # Initialize reversible network
    if mode.lower() == "photorealistic":
        RevNetwork = RevResNet(
            nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
            in_channel=3, mult=4, hidden_dim=16, sp_steps=2)
    elif mode.lower() == "artistic":
        RevNetwork = RevResNet(
            nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
            in_channel=3, mult=4, hidden_dim=64, sp_steps=1)


    state_dict = torch.load(ckpt_path, map_location=device)
    RevNetwork.load_state_dict(state_dict['state_dict'])
    RevNetwork = RevNetwork.to(device)
    RevNetwork.eval()

    cwct = cWCT()

    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)

    with torch.no_grad():
        # Encode
        z_c = RevNetwork(content_tensor, forward=True)
        z_s = RevNetwork(style_tensor, forward=True)

        # Transfer
        if alpha_c is not None and content_seg is None and style_seg is None:
            assert 0.0 <= alpha_c <= 1.0
            z_cs = cwct.interpolation(z_c, styl_feat_list=[z_s], alpha_s_list=[1.0], alpha_c=alpha_c)
        else:
            z_cs = cwct.transfer(z_c, z_s, content_seg, style_seg)

        # Decode
        stylized_tensor = RevNetwork(z_cs, forward=False)

    # img_tensor = ((stylized_tensor.detach().cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # img_tensor = img_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    # cv2.imwrite("style.png", img_tensor[0][...,[2,1,0]])
        
    # img_tensor = ((content_tensor.detach().cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # img_tensor = img_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    # cv2.imwrite("content_tensor.png", img_tensor[0][...,[2,1,0]])
    # img_tensor = ((style_tensor.detach().cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # img_tensor = img_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    # cv2.imwrite("style_tensor.png", img_tensor[0][...,[2,1,0]])

    # img_tensor = ((stylized_tensor.detach().cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # img_tensor = img_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    # cv2.imwrite("stylized_tensor.png", img_tensor[0][...,[2,1,0]])

    return stylized_tensor  # shape: (1, 3, H, W)
