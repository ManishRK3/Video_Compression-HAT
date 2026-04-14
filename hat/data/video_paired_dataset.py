import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from hat.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VideoPairedDataset(Dataset):
    """
    Iterates over video files frame-by-frame, returning paired LR/HR tensors.

    Each item in the dataset represents one frame. The dataset scans all .mp4
    and .avi files under `dataroot_hr` at init time and records (video_path,
    frame_idx) references — actual frame decoding happens in __getitem__.

    Expected keys in opt:
        dataroot_hr (str):  Root directory containing video files.
        scale       (int):  Downscale factor for LR generation. Default: 2.
        hr_height   (int):  Target HR height. Default: 360.
        hr_width    (int):  Target HR width.  Default: 640.
        fps         (int):  Frame rate used when writing output videos.
                            Not used by the dataset itself; stored here so the
                            test pipeline can read it from one place. Default: 30.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = opt.get('scale', 2)
        hr_h = opt.get('hr_height', 360)
        hr_w = opt.get('hr_width', 640)
        lr_h, lr_w = hr_h // self.scale, hr_w // self.scale

        self.lr_transform = transforms.Compose([
            transforms.Resize((lr_h, lr_w), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_h, hr_w), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        # Build index: list of (video_path, frame_idx) for every frame in
        # every video found under dataroot_hr.
        # Also record each video's native FPS so the test pipeline can use it
        # when writing output video files — no need to hardcode it in the config.
        self.frame_refs = []
        self.video_fps = {}  # video_name -> fps
        for root, _, files in os.walk(opt['dataroot_hr']):
            for fname in sorted(files):
                if not fname.endswith(('.mp4', '.avi')):
                    continue
                video_path = os.path.join(root, fname)
                cap = cv2.VideoCapture(video_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                video_name = os.path.splitext(fname)[0]
                self.video_fps[video_name] = fps
                self.frame_refs.extend(
                    (video_path, i) for i in range(n_frames)
                )

    def __len__(self):
        return len(self.frame_refs)

    def __getitem__(self, idx):
        video_path, frame_idx = self.frame_refs[idx]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(
                f"Failed to read frame {frame_idx} from {video_path}"
            )

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lr_img = self.lr_transform(img)
        hr_img = self.hr_transform(img)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        return {
            'lq': lr_img,
            'gt': hr_img,
            'video_name': video_name,
            'frame_idx': frame_idx,
            # lq_path is a dummy string so basicsr metrics code doesn't break
            # if it falls back to image-style validation.
            'lq_path': f"{video_name}_frame{frame_idx:04d}.png",
        }
