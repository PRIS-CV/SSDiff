import numpy as np
# from torchvision.models import resnet50
import torch
from torch.backends import cudnn
import tqdm
from thop import profile
from thop import clever_format

import guided_diffusion.face_parsing.bisenet as bisenet
import guided_diffusion.codeformer.codeformer_arch as codeformer_arch


if __name__ == '__main__':
    # use fvcore for flops accounting
    # from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

    height = 256
    width = 256
    
    model = codeformer_arch.CodeFormer()

    input = torch.randn(1, 3, height, width)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = input.to(device)
    macs, params = profile(model.to(device), inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print("macs:", macs)
    print("params:", params)


    # cudnn.benchmark = True

    # device = 'cpu:0' #'cuda:0'

    # repetitions = 10

    # model = model.to(device)
    # dummy_input = torch.randn(1, 3, height, width).to(device)

    # # synchronize / wait for all the GPU process then back to cpu
    # torch.cuda.synchronize()

    # # testing CUDA Event
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # # initialize
    # timings = np.zeros((repetitions, 1))

    # print('testing ...\n')
    # with torch.no_grad():
    #     for rep in tqdm.tqdm(range(repetitions)):
    #         starter.record()
    #         _ = model(dummy_input)[0]
    #         ender.record()
    #         torch.cuda.synchronize()  # wait for ending
    #         curr_time = starter.elapsed_time(ender)  # from starter to ender (/ms)
    #         timings[rep] = curr_time

    # avg = timings.sum() / repetitions
    # print('\navg={}\n'.format(avg))