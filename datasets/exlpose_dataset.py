import os

import cv2
import einops
import torch
from crowdposetools.coco import COCO
from torch.utils.data import Dataset

EXLPOSE_ROOT = "/workspace/ExLPose"


class ExlposeDataset(Dataset):
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
        return img

    def __getitem__(self, index):
        res = self.get_image(index)
        res = einops.rearrange(res, "h w c -> c h w")
        res = torch.from_numpy(res).float() / 255.0

        return res, res.clone()

    def __len__(self):
        return len(self.img_ids)
