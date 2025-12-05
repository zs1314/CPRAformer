import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from data_RGB import get_test_data
from skimage import img_as_ubyte
from pdb import set_trace as stx
from layers import *

parser = argparse.ArgumentParser(description='Image Deraining using DAIT')

# 测试集的输入，供载入权重后的模型推理
parser.add_argument('--input_dir', default='./Datasets/Synthetic_Rain_Datasets/test/', type=str, help='Directory of validation images')
# parser.add_argument('--input_dir', default='/root/autodl-tmp/real_test_1000', type=str, help='Directory of validation images')
# parser.add_argument('--input_dir', default='/root/autodl-tmp/raindrop/test_b', type=str, help='Directory of validation images')

# 设置推理得到的图片存放的地址，用于后续与真实的图片计算相关指标（evaluate_PSNR_SSIM.m）
parser.add_argument('--result_dir', default='/root/autodl-fs/results', type=str, help='Directory for results')
# 设置推理时加载的权重
parser.add_argument('--weights', default='/root/autodl-fs/checkpoints_Rain200L/Deraining/models/DAIT/model_best.pth',
                    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--model', default='DAIT', type=str, help='Directory for model results')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
from models import DAIT

model_restoration = DAIT.DAIT()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()
win = 128  # window size
datasets = ['Test2800', 'Test1200', 'Rain100L', 'Rain100H', 'Test100']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset, 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, args.model)
    result_dir = os.path.join(result_dir, dataset)
    utils.mkdir(result_dir)

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]
            _, _, Hx, Wx = input_.shape
            input_re, batch_list = window_partitionx(input_, win)
            restored = model_restoration(input_re)
            restored = window_reversex(restored[0], win, Hx, Wx, batch_list)

            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), restored_img)

# Shut down automatically after training completion.
# 这个命令有毒，建议可选
# os.system("/usr/bin/shutdown")
