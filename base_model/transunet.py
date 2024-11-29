import torch
import numpy as np
from base_model.vit_seg_modeling import VisionTransformer as ViT_seg
from base_model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def transNet(n_classes, img_size=512):
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)
    return net


if __name__ == '__main__':
    import time
    from thop import profile
    model = transNet(2).cuda()
    x = torch.randn(1, 3, 512, 512).cuda()
    segments = model(x)
    num_runs = 10
    total_time = 0
    # 多次推理，计算平均推理时间
    for _ in range(num_runs):
        start_time = time.time()
        results = model(x)
        end_time = time.time()
        total_time += (end_time - start_time)
    # 计算平均推理时间
    avg_inference_time = total_time / num_runs
    # 计算FPS
    fps = 1 / avg_inference_time
    print(f"FPS: {fps:.2f} frames per second")
    flops, params = profile(model, inputs=(x,))
    print(f'FLOPs: {flops / 1e9}G')
    print(f'params: {params / 1e6}M')
    print(segments.size())