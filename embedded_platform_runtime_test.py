import argparse
import os
import time

import tensorrt as trt
import torch
import torchvision
from networks.RTMonoDepth.RTMonoDepth import RTMonoDepth
from torch2trt import torch2trt


class Inference_Engine():
    def __init__(self, args):

        h,w = args.shape

        mode = "cpu" if args.cpu else "cuda"

        self.input_t1 = torch.randn(1, 3, h, w).to(mode)

        print("Loading pytorch models...")
        self.model = RTMonoDepth().cuda()


        self.model_trt, _ = self.convert_PyTorch_to_TensorRT(self.model)    

        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        for i in range(args.cycles + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            self.input_t1 = torch.randn(1, 3, h, w).to(mode)

            torch.cuda.synchronize() #Synchronize transfer to cuda
            t0 = time.time()
            out = self.model_trt(self.input_t1)
            torch.cuda.synchronize()
            
            times += time.time() - t0

        times = times / args.cycles
        fps = 1 / times
        print('[tensorRT] Runtime: {}s'.format(times))
        print('[tensorRT] FPS: {}\n'.format(fps))



    def convert_PyTorch_to_TensorRT(self, model):
        print('[tensorRT] Starting TensorRT conversion')
        model_trt = torch2trt(model, [self.input_t1], fp16_mode=True)
        print("[tensorRT] Model converted to TensorRT")

        torch.save(model_trt.state_dict(), 'trt_model.pth')
        print("model saved.")

        from torch2trt import TRTModule
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load('trt_model.pth'))
        print("model loaded.")

        return model_trt, None #engine


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--shape', type=int, nargs='+', default=[192,640], help="[H,W]")
    parser.add_argument('--cycles', type=int, default=200)
    parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')

    args = parser.parse_args()

    engine = Inference_Engine(args)
