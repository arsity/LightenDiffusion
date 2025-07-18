import os

import cv2
import einops
import torch
import torch.utils.data
from crowdposetools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset

from datasets.data_augment import PairCompose, PairRandomHorizontalFilp, PairToTensor

EXLPOSE_ROOT = "/workspace/ExLPose"


class ExlPoseDataset(Dataset):
    def __init__(
        self, dir, patch_size=512, filelist=None, train=False, split="exlpose"
    ):
        assert not train, "ExlposeDataset is only for evaluation"
        assert split in ["exlpose", "a7m3", "ricoh3"], f"{split=}"

        self.patch_size = patch_size

        if split == "exlpose":
            self.coco = COCO(
                os.path.join(EXLPOSE_ROOT, "Annotations", "ExLPose_test_LL-A.json")
            )
        elif split == "a7m3":
            self.coco = COCO(
                os.path.join(EXLPOSE_ROOT, "Annotations", "ExLPose-OC_test_A7M3.json")
            )
        elif split == "ricoh3":
            self.coco = COCO(
                os.path.join(EXLPOSE_ROOT, "Annotations", "ExLPose-OC_test_RICOH3.json")
            )

        self.img_ids = sorted(self.coco.getImgIds())

    def get_image(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(EXLPOSE_ROOT, img_info["file_name"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        return img, f"{img_id}.png"

    def __getitem__(self, index):
        res, img_id = self.get_image(index)
        res = einops.rearrange(res, "h w c -> c h w")
        res = torch.from_numpy(res).float() / 255.0
        res = torch.cat([res, res.clone()], dim=0)

        return res, img_id

    def __len__(self):
        return len(self.img_ids)


class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        # train_dataset = AllWeatherDataset(
        #     self.config.data.data_dir,
        #     patch_size=self.config.data.patch_size,
        #     filelist="{}_train.txt".format(self.config.data.train_dataset),
        # )
        # val_dataset = AllWeatherDataset(
        #     self.config.data.data_dir,
        #     patch_size=self.config.data.patch_size,
        #     filelist="{}_val.txt".format(self.config.data.val_dataset),
        #     train=False,
        # )
        val_dataset = ExlPoseDataset(
            self.config.data.data_dir,
            patch_size=self.config.data.patch_size,
            filelist="{}_val.txt".format(self.config.data.val_dataset),
            train=False,
        )

        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=self.config.training.batch_size,
        #     shuffle=True,
        #     num_workers=self.config.data.num_workers,
        #     pin_memory=True,
        # )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

        return None, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.file_list = filelist
        self.train_list = os.path.join(dir, self.file_list)
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.patch_size = patch_size

        if train:
            self.transforms = PairCompose([PairRandomHorizontalFilp(), PairToTensor()])
        else:
            self.transforms = PairCompose([PairToTensor()])

    def get_images(self, index):
        input_name = self.input_names[index].replace("\n", "")

        low_img_name, high_img_name = input_name.split(" ")[0], input_name.split(" ")[1]

        img_id = low_img_name.split("/")[-1]
        low_img, high_img = Image.open(low_img_name), Image.open(high_img_name)

        low_img, high_img = self.transforms(low_img, high_img)

        return torch.cat([low_img, high_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
