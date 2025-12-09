import argparse
import os
import cv2
import os.path as osp
from collections import OrderedDict
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from PIL import Image
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import models, transforms
import torch.nn as nn

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    SUPPORTED_TASKS,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_restorer_code,
    create_face_pasring,
    add_dict_to_argparser,
    args_to_dict,
    create_arcface_embedding,
    get_region_mask_3ch,
    expand_mask_vertically,
    compute_pixel_diff,
)


def main():
    def partial_guidance(x, t, pred_ref=None, y=None, pred_xstart=None, target=None, ref=None, mask_o=None, mask=None, mask_select=None, mask_style=None, task="restoration", scale=0, N=1, T1=400, s_start=1, s_end=0.7, restorer_y=None, restorer_y_parsing=None, ref_color_img=None): 
        assert y is not None
        with th.enable_grad():
            pred_xstart_in = pred_xstart.detach().requires_grad_(True)

            total_loss = 0
            print(f'[t={str(t.cpu().numpy()[0]).zfill(3)}]', end=' ')

            # Only blurred
            if "restoration" in task:
                if target == None:
                    fake_g_output = restorer_y                  
                    fake_g_output = fake_g_output.detach().requires_grad_(True).cuda()
                else:
                    fake_g_output = target.detach().requires_grad_(True).cuda()
                if task == "restoration":
                    mse_loss = F.mse_loss(fake_g_output[mask_o == 0], pred_xstart_in[mask_o == 0], reduction='sum') * args.ss_weight # 1
                    print(f'loss (smooth semantics): {mse_loss};', end=' ')
                    total_loss = total_loss + mse_loss


            # Damaged, faded and blurred
            if task == "old_photo_restoration":
                total_loss = 0
                pred_xstart_in = pred_xstart.detach().requires_grad_(True)
                fake_g_output = fake_g_output.detach().requires_grad_(True)

                mse_loss = F.mse_loss((fake_g_output)[mask_o == 0], (pred_xstart_in)[mask_o == 0], reduction='sum')
                print(f'loss (lightness): {mse_loss};', end=' ')
                total_loss = total_loss + mse_loss

                if pred_ref is None:   
                    pred_guided = pred_xstart * mask + pred_xstart.detach() * (1 - mask)
                    ref_guided = ref_color_img * mask + ref_color_img.detach() * (1 - mask)

                    diff_pred = compute_pixel_diff(pred_guided, mask)
                    diff_ref = compute_pixel_diff(ref_guided, mask)

                    ref_loss = F.mse_loss(
                        diff_pred[mask[:, :1, :, :].bool()],
                        diff_ref[mask[:, :1, :, :].bool()],
                        reduction='sum' 
                    ) * 100
                    print(f'loss (ref): {ref_loss};', end=' ')

                    total_loss = total_loss + ref_loss

                if not pred_ref is None:
                    adain_loss1 = F.mse_loss((pred_xstart_in)[mask_style == 0], (pred_ref)[mask_style == 0], reduction='sum') * args.op_color_weight
                    print(f'loss (color1): {adain_loss1};', end=' ')
                    total_loss = total_loss + adain_loss1
                

            if t.cpu().numpy()[0] > 0:
                print(end='\r')
            else:
                print('\n')
            gradient = th.autograd.grad(total_loss, pred_xstart_in)[0]
            # if args.task == "old_photo_restoration":
            #     gradient[mask_o>0] = 0
        if task == "old_photo_restoration":
            return gradient, fake_g_output.detach(), pred_xstart_in, ref_color_img, mask_style
        else:
            return gradient, fake_g_output.detach()
        
    

    def model_fn(x, t, pred_ref=None, y=None, target=None, ref=None, mask_o=None, mask=None, mask_style=None, mask_select=None, task=None, scale=0, N=1, T1=400, s_start=1, s_end=0.7, restorer_y=None, restorer_y_parsing=None, ref_color_img=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = f'{args.out_dir}/s{args.guidance_scale}-seed{args.seed}'
    logger.configure(dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict({key[7:]:value for key, value in state_dict.items()})
    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())
    model.eval()


    # if 'restoration' in args.task:
    logger.log("Loading restorer for codebook...")
    restorer_code = create_restorer_code()
    restorer_code.load_state_dict(
        dist_util.load_state_dict(args.restorer_code_path, map_location="cpu")['params_ema'], strict=False
    )
    restorer_code.to(dist_util.dev())
    restorer_code.eval()

    logger.log("Loading face pasring prediction...")
    face_pasring = create_face_pasring()
    face_pasring.load_state_dict(
        dist_util.load_state_dict(args.face_pasring_path, map_location="cpu"), strict=False
    )
    face_pasring.to(dist_util.dev())
    face_pasring.eval()

    
    assert args.task in SUPPORTED_TASKS, "Task not supported!"

    print("=================== Summary (Sampling) ===================")

    print(f'Task: {args.task}; Guidance scale: {args.guidance_scale}')
    if args.N > 1:
        print(f'From {args.s_start}T to {args.s_end}T, {args.N} gradient steps are taken at each time step.')

    if args.task == 'restoration':
        print(f'Apply partial guidance on smooth semantics (w={args.ss_weight}).')
    elif args.task == 'old_photo_restoration':
        print(f'Apply partial guidance on old photo lightness (w={args.op_lightness_weight}).')
        print(f'Apply partial guidance on old photo color stats (w={args.op_color_weight}).')
    
    print("==========================================================")
    
    

    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    
    all_images = []
    lr_folder = args.in_dir
    lr_images = sorted(os.listdir(lr_folder))

    if args.task == 'old_photo_restoration':
        mask_folder = args.mask_dir
        if mask_folder is None:
            print(f'No mask is inputted!')

    logger.log("Sampling...")

    for img_name in lr_images:

        model_kwargs = {}
        model_kwargs["task"] = args.task
        model_kwargs["target"] = None
        model_kwargs["scale"] = args.guidance_scale
        model_kwargs["N"] = args.N
        model_kwargs["T1"] = args.T1
        model_kwargs["s_start"] = int(args.s_start * args.diffusion_steps)
        model_kwargs["s_end"] = int(args.s_end * args.diffusion_steps)
        y0 = cv2.resize(cv2.imread(osp.join(lr_folder, img_name)), (512,512)).astype(np.float32)[:, :, [2, 1, 0]]/ 127.5 - 1
        model_kwargs["y"] = th.tensor(y0).permute(2,0,1).unsqueeze(0).cuda() # (B,C,H,W), [-1,1]

        restorer_y = restorer_code(model_kwargs["y"], w=args.w).clamp(-1,1)
        model_kwargs["restorer_y"] = restorer_y
        restorer_y_parsing = face_pasring(restorer_y)[0]

        model_kwargs["restorer_y_parsing"] = restorer_y_parsing
        region_mask_style = get_region_mask_3ch(restorer_y_parsing, mask_labels=[0, 16, 17])   # For style transfer, only face+neck
        model_kwargs["mask_style"] = region_mask_style

        model_kwargs["mask_select"] = get_region_mask_3ch(restorer_y_parsing, mask_labels=[11,12,13])   # For aba

        
        if 'old_photo_restoration' in args.task:
            try:
                ref_color_img = cv2.resize(cv2.imread(osp.join(args.self_dir, img_name)), (512,512)).astype(np.float32)[:, :, [2, 1, 0]]/ 127.5 - 1
                model_kwargs["ref_color_img"] = th.tensor(ref_color_img).permute(2,0,1).unsqueeze(0).cuda() # (B,C,H,W), [0,1]

                mask_img = cv2.resize(cv2.imread(osp.join(mask_folder, img_name)), (512,512)).astype(np.float32)/ 255.
                model_kwargs["mask_o"] = th.tensor(mask_img).permute(2,0,1).unsqueeze(0).cuda() # (B,C,H,W), [0,1]

                # Make masks larger except for facial components area in "mask_o"
                region_mask = get_region_mask_3ch(restorer_y_parsing, mask_labels=[0,1,14,17])   # background + skin -> w/o facial components
                mask_o_expand = (region_mask > 0.5) & (expand_mask_vertically(model_kwargs["mask_o"], expand_y=4, expand_x=4) > 0.5) # del facial components parts in mask_o
                model_kwargs["mask_o"] = (mask_o_expand > 0.5) | (model_kwargs["mask_o"] > 0.5) 

                # for 
                region_mask_3ch = get_region_mask_3ch(restorer_y_parsing, mask_labels=[0])   # For restoration 0, 1, 17
                region_mask_3ch_bool = (region_mask_3ch > 0.5) 
                mask_o_bool = (model_kwargs["mask_o"] > 0.5) 
                model_kwargs["mask"] = expand_mask_vertically((region_mask_3ch_bool & mask_o_bool))

            except:
                print('Warning: Will treat as if there are no missing pixels!')
                mask_img = np.zeros((512, 512, 3)).astype(np.float32)/ 255.
                model_kwargs["mask_o"] = th.tensor(mask_img).permute(2,0,1).unsqueeze(0).cuda() # (B,C,H,W), [0,1]

        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=partial_guidance,
            device=dist_util.dev(),
            seed=seed
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} sample")

        cv2.imwrite(f'{out_dir}/{img_name}', all_images[-1][0][...,[2,1,0]])

    dist.barrier()
    logger.log("Sampling complete!")



def create_argparser():
    defaults = dict(
        seed=1234,
        task='restoration',
        in_dir='testdata/cropped_faces',
        out_dir='results/blind_restoration',
        self_dir='results/old_photo_face/ref/hard/s0.001-seed4321/',
        ref_dir=None,
        mask_dir=None,
        w = 0.5,
        lightness_weight=1.0,
        color_weight=0.05,
        unmasked_weight=1.0,
        ss_weight=1.0,
        ref_weight=25.0,
        op_lightness_weight=1.0,
        op_color_weight=20.0,
        N=1,                        # number of gradient steps at each time t
        T1=400,                     # stage for Selective Coloring
        s_start=1.0,                # range for multiple gradient steps (S_{start} = s_start * T)
        s_end=0.7,                  # range for multiple gradient steps (S_{end} = s_end * T)
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="models/iddpm_ffhq512_ema500000.pth",
        restorer_code_path="models/restorer/codeformer.pth",
        face_pasring_path="models/face_parsing/resnet34.pt",
        guidance_scale=0.1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()