import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class RealVideoSingleImageDataset(Dataset):
    """
    For each frame in HR videos, generates a LR frame on-the-fly and returns both.
    Suitable for real-world video SR inference where only HR videos are available.

    Args in opt:
        dataroot_hr (str): Directory containing HR video files.
        scale (int): Downscale factor for LR. Default: 4.
        hr_height (int): HR frame height. Optional.
        hr_width (int): HR frame width. Optional.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = opt.get('scale', 4)
        self.dataroot_hr = opt['dataroot_hr']
        self.hr_h = opt.get('hr_height', None)
        self.hr_w = opt.get('hr_width', None)

        self.frame_refs = []
        self.video_fps = {}  # video_name -> fps
        for root, _, files in os.walk(self.dataroot_hr):
            for fname in sorted(files):
                if not fname.endswith((".mp4", ".avi")):
                    continue
                video_path = os.path.join(root, fname)
                cap = cv2.VideoCapture(video_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                video_name = os.path.splitext(fname)[0]
                self.video_fps[video_name] = fps
                for i in range(n_frames):
                    self.frame_refs.append((video_path, i))

        self.hr_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.frame_refs)

    def __getitem__(self, idx):
        video_path, frame_idx = self.frame_refs[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.hr_h and self.hr_w:
            img = img.resize((self.hr_w, self.hr_h), Image.BICUBIC)
        hr_img = self.hr_transform(img)
        # Always downscale by 2x for input to model
        lr_h = hr_img.shape[1] // 2
        lr_w = hr_img.shape[2] // 2
        lr_img_pil = transforms.functional.resize(img, (lr_h, lr_w), interpolation=Image.BICUBIC)
        lr_img = self.hr_transform(lr_img_pil)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        return {
            'lq': lr_img,
            'gt': hr_img,
            'video_name': video_name,
            'frame_idx': frame_idx,
            'lq_path': f"{video_name}_frame{frame_idx:04d}.png",
        }
