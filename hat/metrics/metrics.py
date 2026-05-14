import cv2
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve
from scipy.special import gamma

from hat.utils.image_util import bgr2ycbcr, rgb2ycbcr_pt
from hat.utils.registry import METRIC_REGISTRY


# -------------------------------------------------------------------- #
# -------------------------- metric utilities ------------------------- #
# -------------------------------------------------------------------- #

def reorder_image(img, input_order='HWC'):
    """Reorder image to HWC. Handles (h,w), (h,w,c), (c,h,w)."""
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Convert [0,255] BGR image to Y channel (float, [0,255])."""
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


# -------------------------------------------------------------------- #
# ------------------------------ PSNR/SSIM --------------------------- #
# -------------------------------------------------------------------- #

@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """PSNR. img/img2: ndarray [0,255]."""
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)


@METRIC_REGISTRY.register()
def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """PSNR (PyTorch). img/img2: tensor (n,3/1,h,w) [0,1]."""
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)
    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)
    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


@METRIC_REGISTRY.register()
def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """SSIM. img/img2: ndarray [0,255]."""
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    ssims = [_ssim(img[..., i], img2[..., i]) for i in range(img.shape[2])]
    return np.array(ssims).mean()


@METRIC_REGISTRY.register()
def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """SSIM (PyTorch). img/img2: tensor (n,3/1,h,w) [0,1]."""
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)
    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)
    return _ssim_pth(img * 255., img2 * 255.)


def _ssim(img, img2):
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def _ssim_pth(img, img2):
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = torch.from_numpy(np.outer(kernel, kernel.transpose())).view(1, 1, 11, 11).expand(
        img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


# -------------------------------------------------------------------- #
# -------------------------------- NIQE ------------------------------ #
# -------------------------------------------------------------------- #

def _estimate_aggd_param(block):
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return alpha, beta_l, beta_r


def _compute_niqe_feature(block):
    feat = []
    alpha, beta_l, beta_r = _estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])
    for shift in [[0, 1], [1, 0], [1, 1], [1, -1]]:
        shifted_block = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = _estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def _niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    assert img.ndim == 2
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []
    for scale in (1, 2):
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        img_normalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_normalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                                       idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(_compute_niqe_feature(block))
        distparam.append(np.array(feat))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True) * 255.

    distparam = np.concatenate(distparam, axis=1)
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(np.matmul((mu_pris_param - mu_distparam), invcov_param),
                        np.transpose((mu_pris_param - mu_distparam)))
    return float(np.squeeze(np.sqrt(quality)))


@METRIC_REGISTRY.register()
def calculate_niqe(img, crop_border, input_order='HWC', convert_to='y', **kwargs):
    """NIQE. img: ndarray [0,255] BGR."""
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    niqe_pris_params = np.load(os.path.join(ROOT_DIR, 'niqe_pris_params.npz'))
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    return _niqe(img.round(), mu_pris_param, cov_pris_param, gaussian_window)


# keep imresize accessible in this module (used by _niqe)
from hat.utils.image_util import imresize  # noqa: E402
