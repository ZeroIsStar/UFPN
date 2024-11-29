import base_model
from torch import nn
from torch import optim
from loss import TL_loss, focal_hausdorffErloss, Lovasz_ce_loss, DynamicWeightedCrossEntropyLoss,FocalLoss, DynamicFocalLoss
from loss.lovasz import LovaszSoftmaxLoss
from Data_loader import cfg


class Loader(dict):
    def __init__(self, model_type = None):
        self.cfg = cfg
        self.model = model_type
        self.model = self.get_segment_model()
        self.loss_function = self.get_loss_function()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_lr_scheduler()

    def get_segment_model(self):
        # 模型
        Model = None
        if self.model == 'Unet':
            Model = base_model.UNet(n_channels=self.cfg.dataset.in_channels, n_classes=self.cfg.dataset.Class, bilinear=True)
        if self.model == 'Unet++':
            Model = base_model.UNet_Nested(n_classes=self.cfg.dataset.Class)
        elif self.model == 'FCN8s':
            Model = base_model.FCN8s(num_classes=self.cfg.dataset.Class)
                elif self.model == 'Deeplabv3+':
            Model = base_model.DeepLabV3plus(in_channels=3, num_classes=self.cfg.dataset.Class)
        elif self.model == 'TransUnet':
            Model = base_model.transNet(n_classes=self.cfg.dataset.Class)
        elif self.model == 'A2FPN':
            Model = base_model.A2FPN(band=3,class_num=self.cfg.dataset.Class)
        elif self.model == 'CMTFNet':
            Model = base_model.CMTFNet(num_classes=self.cfg.dataset.Class)
        elif self.model == 'EfficientPyramidMamba':
            Model = base_model.EfficientPyramidMamba()
        elif self.model == 'UFPN':
            Model = base_model.model(3, num_classes=self.cfg.dataset.Class)
        if Model is None:
            raise ValueError('未知的模型配置：' + str(self.model))
        return Model

    def get_loss_function(self):
        # 初始化为None或默认损失函数
        Loss_Function = None
        # 损失函数
        if self.cfg.train.loss_function == 'celoss':
            Loss_Function = nn.CrossEntropyLoss(weight=self.cfg.train.loss_function_weight)
        elif self.cfg.train.loss_function == 'Tversky_loss_lovasz':
            Loss_Function = TL_loss(alpha=0.5, beta=0.5, n_class = self.cfg.dataset.Class)
        elif self.cfg.train.loss_function == 'f-h-loss':
            Loss_Function = focal_hausdorffErloss()
        elif self.cfg.train.loss_function == 'lovasz_ce_loss':
            Loss_Function = Lovasz_ce_loss(weight=self.cfg.train.loss_function_weight, n_class=self.cfg.dataset.Class)
        elif self.cfg.train.loss_function == 'lovasz_softmax':
            Loss_Function = LovaszSoftmaxLoss
        elif self.cfg.train.loss_function == 'DynamicWeightedCrossEntropyLoss':
            Loss_Function = DynamicWeightedCrossEntropyLoss()
        elif self.cfg.train.loss_function == 'focalloss':
            Loss_Function = FocalLoss()
        elif self.cfg.train.loss_function == 'dynamic_focal_loss':
            Loss_Function = DynamicFocalLoss()
        if Loss_Function is None:
            raise ValueError("未知的损失函数配置：" + self.cfg.train.loss_function)
        return Loss_Function

    def get_optimizer(self):
        # 优化器
        if self.cfg.optimizer.type == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), momentum=self.cfg.optimizer.momentum, lr=self.cfg.optimizer.base_lr,
                                  weight_decay=self.cfg.optimizer.weight_decay)  # 优化器
        elif self.cfg.optimizer.type == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.optimizer.base_lr, betas=(0.9, 0.999), weight_decay=0.01)
        else:
            return NotImplementedError('未应用', self.cfg.optimizer.type)
        return optimizer

    def get_lr_scheduler(self):
        if self.cfg.scheduler.type == 'Poly':
            def lambda_rule(epoch):
                # 定义预热步数和总训练步数
                num_warmup_steps = self.cfg.scheduler.epoch*0.15
                num_training_steps = self.cfg.scheduler.epoch
                if epoch < num_warmup_steps:
                    return float(epoch) / float(max(1, num_warmup_steps))
                else:
                    return max(0.0, float(num_training_steps - epoch) / float(
                        max(1, num_training_steps - num_warmup_steps)))
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        elif self.cfg.scheduler.type == 'step':
            step_size = self.cfg.scheduler.epoch // 3
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)
        elif self.cfg.scheduler.type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                             T_max=self.cfg.scheduler.epoch,
                                                             eta_min=self.cfg.optimizer.min_lr)
        else:
            return NotImplementedError('未应用', self.cfg.scheduler.type)
        return scheduler









