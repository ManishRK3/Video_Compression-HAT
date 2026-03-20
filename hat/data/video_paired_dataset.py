import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class VideoPairedDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dataroot_hr = opt['dataroot_hr']
        self.scale = opt.get('scale', 2)
        self.hr_height = opt.get('hr_height', 360)
        self.hr_width = opt.get('hr_width', 640)
        self.frame_refs = []
        self.lr_transform = transforms.Compose([
            transforms.Resize((int(self.hr_height / self.scale), int(self.hr_width / self.scale)), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        self.hr_transform = transforms.Compose([
            transforms.Resize((self.hr_height, self.hr_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        for root, _, files in os.walk(self.dataroot_hr):
            for f in files:
                if f.endswith('.mp4') or f.endswith('.avi'):
                    video_path = os.path.join(root, f)
                    cap = cv2.VideoCapture(video_path)
                    frame_idx = 0
                    while True:
                        ret, frame = cap.read()
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
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        # Convert frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Apply transforms
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(img)

        # Upscale LR frame using bicubic (for saving upscaled frame)
        upscaled_img = transforms.Resize((self.hr_height, self.hr_width), Image.BICUBIC)(transforms.ToPILImage()(lr_img))
        upscaled_tensor = transforms.ToTensor()(upscaled_img)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        lq_path = f"{video_name}_frame{frame_idx}.png"

        # Save HR, LR, and upscaled frames
        save_dir = self.opt.get('save_dir', None)
        if save_dir:
            hr_save_dir = os.path.join(save_dir, video_name, 'HR')
            lr_save_dir = os.path.join(save_dir, video_name, 'LR')
            upscaled_save_dir = os.path.join(save_dir, video_name, 'UP')
            os.makedirs(hr_save_dir, exist_ok=True)
            os.makedirs(lr_save_dir, exist_ok=True)
            os.makedirs(upscaled_save_dir, exist_ok=True)
            # Save HR
            transforms.ToPILImage()(hr_img).save(os.path.join(hr_save_dir, f'frame_{frame_idx:04d}.png'))
            # Save LR
            transforms.ToPILImage()(lr_img).save(os.path.join(lr_save_dir, f'frame_{frame_idx:04d}.png'))
            # Save upscaled
            upscaled_img.save(os.path.join(upscaled_save_dir, f'frame_{frame_idx:04d}.png'))

        return {'lq': lr_img,'gt': hr_img,'upscaled': upscaled_tensor,'video_name': video_name,'frame_idx': frame_idx}
