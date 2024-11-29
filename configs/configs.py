import torch

cfg = dict(
    segment_model = dict(
        type ='FCN8s',  # ['FCN8s','FCN16s','FCN32s','Unet','Deeplabv3+','SCDUNetPP']
    ),

    dataset = dict(
        set          = ['train', 'val', 'test'],
        dataset_name = 'palu',
        # ['palu','Mengdong','Hokkaido_Iburi_Tobu','Jiuzhai_valley"_(0.2m)','Longxi_River'(SAT),'Moxi_town"_(0.2m)‘] Seg
        batch_size   = 2,
        in_channels=3,
        Class=2,
        # clip_grad_value_ = 5.0  # 模型梯度裁剪
    ),

    optimizer = dict(
        type         = 'SGD',  # ['SGD','AdamW']
        base_lr      = 5e-4,
        min_lr       = 0,
        step_size    = 10,
        gamma        = 0.9,
        weight_decay = 5e-4,
        momentum     = 0.99
    ),
    train = dict(
        loss_function='Tversky_loss_lovasz',  # ['celoss','Tversky_loss_lovasz','f-h-loss', 'lovasz_ce_loss','lovasz_softmax','DynamicWeightedCrossEntropyLoss', 'dynamic_focal_loss']
        loss_function_weight = torch.tensor([1.0, 1.0]),
    ),

    scheduler=dict(
        #['linear', 'step', 'CosineAnnealingLR'] 内置
        type         = 'Poly',
        epoch        = 200,
    )
)
