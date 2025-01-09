import torch.distributed as dist
from scipy.stats import beta
import torch
from typing import Tuple
import matplotlib.pyplot as plot

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    D = size[4]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cut_d = np.int(D * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(D)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_d // 2, 0, D)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)

    return bbx1, bby1,bbz1, bbx2, bby2, bbz2

def mixup_data(x, y, alpha=0.5, index=None, lam=None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if lam is None:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = lam

    lam = max(lam, 1 - lam)
    batch_size = x.size()[0]
    if index is None:
        index = torch.randperm(batch_size).cpu()
        print(index)
    else:
        index = index
    print(x[index, :].shape)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    print(np.unique((1 - lam) * x[index, :]))
    print(np.unique(lam * x))
    print("mixed_x:",mixed_x.shape)
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y, lam, index

def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)

def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)

def plot1():
    # 设置 plot 支持中文
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

    # 定义一组alpha 跟 beta值
    alpha_beta_values = [[0.1, 0.1], [0.2, 0.2], [0.05,0.05], [2, 2], [2, 5]]
    linestyles = []

    # 定义 x 值
    x = np.linspace(0, 1, 1003)[1:-1]

    for alpha_beta_value in alpha_beta_values:
        print(alpha_beta_value)
        dist = beta(alpha_beta_value[0], alpha_beta_value[1])

        dist_y = dist.pdf(x)

        # 添加图例
        # plot.legend('alpha=')
        # 创建 beta 曲线
        plot.plot(x, dist_y, label=r'$\alpha=%.1f,\ \beta=%.1f$' % (alpha_beta_value[0], alpha_beta_value[1]))

    # 设置标题
    plot.title(u'B分布', fontproperties=font)
    # 设置 x,y 轴取值范围
    plot.xlim(0, 1)
    plot.ylim(0, 2.5)
    plot.legend()
    plot.show()


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor

if __name__ == '__main__':
    import torch,os
    import numpy as np
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    bbx1, bby1,bbz1, bbx2, bby2, bbz2 = rand_bbox((2,4,128,128,128),0.5)
    print("bbx1, bby1,bbz1, bbx2, bby2, bbz2",bbx1, bby1,bbz1, bbx2, bby2, bbz2)
    import nibabel as nib
    # path = 'J:\Medical image\BraTS\BraTS2020\mixup'
    # flair_001 = nib.load(os.path.join(path,'BraTS20_Training_001','BraTS20_Training_001_flair.nii.gz'))
    # seg_001 = nib.load(os.path.join(path,'BraTS20_Training_001','BraTS20_Training_001_seg.nii.gz'))
    #
    # flair_003 = nib.load(os.path.join(path,'BraTS20_Training_003','BraTS20_Training_003_flair.nii.gz'))
    # seg_003 = nib.load(os.path.join(path, 'BraTS20_Training_003','BraTS20_Training_003_seg.nii.gz'))
    #
    # t1ce_001_data = flair_001.get_fdata()
    # seg_001_data = seg_001.get_fdata()
    #
    # t1ce_003_data = flair_003.get_fdata()
    # seg_003_data = seg_003.get_fdata()
    # import time
    # start_time = time.time()
    # # mixed_x, mixed_y, lam, index = mixup_data(torch.tensor([t1ce_001_data,t1ce_003_data]),torch.tensor([seg_001_data,seg_003_data]))
    # lam = np.random.beta(0.5, 0.5)
    # print()
    # mixed_x,mixed_y = mixup(torch.tensor([t1ce_001_data,t1ce_003_data]),torch.tensor([seg_001_data,seg_003_data]),lam)
    # end_time = time.time()
    # total_time = (end_time - start_time)
    # print('The total training time is {:.4f} seconds'.format(total_time))
    # mix_image = nib.Nifti1Image(mixed_x[0],affine=flair_001.affine,header=flair_001.header)
    # mix_image.to_filename(os.path.join(path,'BraTS20_Training_001','BraTS20_Training_001_1_flair_{}.nii.gz'.format(lam)))
    #
    # mix_image2 = nib.Nifti1Image(mixed_x[1],affine=flair_001.affine,header=flair_001.header)
    # mix_image2.to_filename(os.path.join(path,'BraTS20_Training_001','BraTS20_Training_001_2_flair_{}.nii.gz'.format(lam)))
    #
    # mix_seg = nib.Nifti1Image(np.array(np.round(mixed_y[0]),dtype=np.int8),affine=flair_001.affine,header=flair_001.header)
    # mix_seg.to_filename(os.path.join(path,'BraTS20_Training_001','BraTS20_Training_001_1_seg_{}.nii.gz'.format(lam)))
    #
    # mix_seg2 = nib.Nifti1Image(np.array(np.round(mixed_y[1]),dtype=np.int8), affine=flair_001.affine, header=flair_001.header)
    # mix_seg2.to_filename(os.path.join(path, 'BraTS20_Training_001', 'BraTS20_Training_001_2_seg_{}.nii.gz'.format(lam)))