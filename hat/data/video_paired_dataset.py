import os
import cv2
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class VideoPairedDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dataroot_hr = opt['dataroot_hr']
        self.scale = opt.get('scale', 2)
        self.frame_refs = []
        for root, _, files in os.walk(self.dataroot_hr):
            for f in files:
                if f.endswith('.mp4') or f.endswith('.avi'):
                    video_path = os.path.join(root, f)
                    cap = cv2.VideoCapture(video_path)
                    frame_idx = 0
                    while True:
                        ret, _ = cap.read()
                        if not ret:
                            break
                        self.frame_refs.append((video_path, frame_idx))
                        frame_idx += 1
                    cap.release()
    def __len__(self):
        return len(self.frame_refs)
    def __getitem__(self, idx):
        video_path, frame_idx = self.frame_refs[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, hr_frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        hr_frame = cv2.resize(hr_frame, (640, 360))
        lr_frame = cv2.resize(hr_frame, (320, 180))
        lr_frame = cv2.resize(lr_frame, (640, 360), interpolation=cv2.INTER_NEAREST)
        hr_tensor = torch.from_numpy(hr_frame).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(lr_frame).permute(2, 0, 1).float() / 255.0
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        lq_path = f"{video_name}_frame{frame_idx}.png"
        return {'lq': lr_tensor, 'gt': hr_tensor, 'lq_path': lq_path}
