import importlib
import torch.utils.data as data_utils
from Data_loader.Dataset import landslide_Dataset
from Data_loader.paper1 import clearing_Dataset


class Struct(dict):
    def __getattr__(self, item):
        try:
            value = self[item]
            if type(value) == type({}):
                return Struct(value)
            return value
        except KeyError:
            raise AttributeError(item)

    def set_cd_cfg_from_file(cfg_path='D:\\detection\\configs\\configs.py'):
        module_spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        cfg = module.cfg
        cfg = Struct(cfg)
        return cfg


cfg = Struct.set_cd_cfg_from_file()


class DataLoader:
    def __init__(self, dataset_name):
        self.dataset       = dataset_name
        self.dataset       = landslide_Dataset
        self.clearing_dataset = clearing_Dataset
        self.Train_dataset = None
        self.Val_dataset   = None
        self.Test_dataset  = None
        if dataset_name == 'palu':
            palu_dir = "E:\\CAS_Landslide\\palu"
            self.Train_dataset = self.dataset(dir=palu_dir, set=cfg.dataset.set[0])
            self.Val_dataset   = self.dataset(dir=palu_dir, set=cfg.dataset.set[1])
            self.Test_dataset  = self.dataset(dir=palu_dir, set=cfg.dataset.set[2])
        elif dataset_name == 'WenChuan':
            Lombok = r'E:\CAS_Landslide\Wenchuan'
            self.Train_dataset = self.dataset(dir=Lombok, set=cfg.dataset.set[0])
            self.Val_dataset   = self.dataset(dir=Lombok, set=cfg.dataset.set[1])
            self.Test_dataset  = self.dataset(dir=Lombok, set=cfg.dataset.set[2])
        elif dataset_name == 'Tiburon_Peninsula（Sentinel）':
            Lombok = r'E:\CAS_Landslide\Tiburon Peninsula（Sentinel）'
            self.Train_dataset = self.dataset(dir=Lombok, set=cfg.dataset.set[0])
            self.Val_dataset   = self.dataset(dir=Lombok, set=cfg.dataset.set[1])
            self.Test_dataset  = self.dataset(dir=Lombok, set=cfg.dataset.set[2]
                                              
    def get_dataloader(self, batch_size=cfg.dataset.batch_size):
        train_loader = data_utils.DataLoader(self.Train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = data_utils.DataLoader(self.Val_dataset, batch_size=batch_size, shuffle=True)
        test_loader  = data_utils.DataLoader(self.Test_dataset, batch_size=1, shuffle=False)
        return train_loader, val_loader, test_loader


# # dataset_list = ['palu', 'Mengdong', 'Hokkaido_Iburi_Tobu', 'Jiuzhai_valley', 'Longxi_River', 'Moxi_town"_(0.2m']
# Data_loader = DataLoader(dataset_name=cfg.dataset.dataset_name)
# Train_loader, Val_loader, Test_loader = Data_loader.get_dataloader()


