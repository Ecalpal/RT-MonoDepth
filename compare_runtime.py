import argparse
import time
import warnings

import torch
from thop import profile

from networks.FastDepth.model import MobileNetSkipAdd
from networks.RTMonoDepth.RTMonoDepth_s import DepthDecoder, DepthEncoder

warnings.filterwarnings("ignore")



def get_time(f, inputs):
    torch.cuda.synchronize()
    start = time.time()
    out = f(inputs)
    torch.cuda.synchronize()
    out_time = time.time()-start
    return out, out_time

def time_benchmark():

    h,w = args.shape
    mode = "cpu" if args.cpu else "cuda"

    input_t1 = torch.randn(1, 3, h, w).to(mode)

    # FastDepth
    fastdepth = MobileNetSkipAdd().to(mode).eval()
    flops1, para_fastdepth = profile(fastdepth, inputs=(input_t1, ), verbose=False)
    print(f'parameters(FastDepth): {para_fastdepth/1e6:.1f}')

    # self
    self_encoder = DepthEncoder().to(mode).eval()
    self_decoder = DepthDecoder(self_encoder.num_ch_enc).to(mode).eval()
    flops1, para_self_encoder = profile(self_encoder, inputs=(input_t1, ), verbose=False)
    flops2, para_self_decoder = profile(self_decoder, inputs=(self_encoder(input_t1), ), verbose=False)
    print(f'parameters(self_Encoder):  {para_self_encoder}')
    print(f'parameters(self_Decoder):  {para_self_decoder}')
    print(f"parameters(self):  {(para_self_encoder+para_self_decoder)/1e6:.1f} M")

    time_fast = 1e-8
    time_self  = 1e-8
    
    for x in range(args.cycles+args.warmup):
        input_t1 = torch.randn(1, 3, h, w).to(mode)

        # fastdepth
        torch.cuda.synchronize()
        start = time.time()
        out = fastdepth(input_t1)
        torch.cuda.synchronize()
        t = time.time()-start

        if x > args.warmup:
            time_fast += t

        # self
        torch.cuda.synchronize()
        start = time.time()
        out = self_decoder(self_encoder(input_t1))
        torch.cuda.synchronize()
        t = time.time()-start

        if x > args.warmup:
            time_self += t

    FPS_fast = args.cycles/(time_fast)
    FPS_self  = args.cycles/(time_self)
    ratio_fps_fast = (FPS_self-FPS_fast)/FPS_fast*100
    flag_fps_fast  = "↑" if ratio_fps_fast>0 else "↓"

    para_fast = para_fastdepth
    para_self  = para_self_encoder+para_self_decoder
    ratio_para_fast = (para_self-para_fast)/para_fast*100
    flag_para_fast = "↑" if ratio_para_fast>0 else "↓"


    print("Test parameters:")
    print(f"Mode:             {mode}")
    print(f"Number of cycles: {args.cycles}\n")
    print(f"Model:            RT-Monodepth           FastDepth")
    print(f"Time elapsed:     {time_self:<6.2f} s               {time_fast:<6.2f} s")
    print(f"FPS:              {FPS_self:<6.1f}     {flag_fps_fast} {ratio_fps_fast:<6.2f}%   {FPS_fast:<6.1f} ")
    print(f"Parameters:       {para_self/1e6:<6.1f} M   {flag_para_fast} {ratio_para_fast:<6.2f}%   {para_fastdepth/1e6:<6.1f} M")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--shape', type=int, nargs='+', default=[192,640], help="[H,W]")
    parser.add_argument('--cycles', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')

    args = parser.parse_args()

    time_benchmark()
