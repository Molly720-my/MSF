import os
import argparse

import torch
import cv2
import numpy as np
import yaml

from models import model_rrdb, model_swinir
import srdata_test
from torch.utils import data

import logging
from utils import utils_logger, util_calculate_psnr_ssim
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import lpips
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import torch.nn.functional as F
# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex').to('cuda')
def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--opt', type=str, default='options/test_rrdb_P+MSF.yml',help='path to option file', required=False)
    parser.add_argument('--output_path', type=str, default='output',help='path to your output', required=False)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    

    # Initialize MS-SSIM model
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
    # Initialization
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)
        print("ok")
    opt['name'] = opt['name'].replace('RRDB', opt['model_type'])
    print(opt)

    ckpt_path = opt['ckpt_path']

    weight = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    # Models
    if opt['model_type'].lower() == 'rrdb':
        model = model_rrdb.RRDBNet(**opt['model']['rrdb']).to('cuda')
    elif opt['model_type'].lower() == 'swinir':
        model = model_swinir.SwinIR(**opt['model']['swinir']).to('cuda')
    else:
        raise ValueError(f"Model {opt['model_type']} is currently unsupported!")

    model.load_state_dict(weight)
    model = model.cuda()

    # Datasets
    testset = srdata_test.Test(**opt['test'])
    data_loader_test = data.DataLoader(
        testset, 
        **opt['dataloader']['test'],
        shuffle=False,
    )

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    if opt['test']['use_hr']:
        logger_name = opt['stage']
        utils_logger.logger_info(logger_name, os.path.join(args.output_path, logger_name+'.log'), mode='w')
        logger = logging.getLogger(logger_name)
        p = 0
        s = 0
        lpips_sum = 0
        ms_ssim_sum =0
        rmse_sum = 0

        count = 0

    # Start testing
    model.eval()
    for batch in data_loader_test:
        lr = batch['lr']
        fn = batch['fn'][0]
        if opt['test']['use_hr']:
            hr = batch['hr']

        lr = lr.to('cuda')
        with torch.no_grad():
            sr = model(lr)
        sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
        sr = sr * 255.
        sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_path, fn), sr)
        
        if opt['test']['use_hr']:
            hr = hr.squeeze(0).numpy().transpose(1, 2, 0)
            hr = hr * 255.
            hr = np.clip(hr.round(), 0, 255).astype(np.uint8)
            hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
            
            psnr = util_calculate_psnr_ssim.calculate_psnr(sr, hr, crop_border=4, test_y_channel=True)
            ssim = util_calculate_psnr_ssim.calculate_ssim(sr, hr, crop_border=4, test_y_channel=True)
             # Prepare tensors for LPIPS calculation
            sr_tensor = torch.tensor(sr).permute(2, 0, 1).unsqueeze(0).float().div(255).to('cuda')
            hr_tensor = torch.tensor(hr).permute(2, 0, 1).unsqueeze(0).float().div(255).to('cuda')
            lpips_value = lpips_model(sr_tensor, hr_tensor).item()

            ms_ssim_value = ms_ssim(sr_tensor, hr_tensor).item()  # Compute MS-SSIM
            rmse_value = torch.sqrt(F.mse_loss(sr_tensor, hr_tensor)).item()  # Compute RMSE            

            p += psnr
            s += ssim
            lpips_sum += lpips_value  # Accumulate LPIPS
            ms_ssim_sum += ms_ssim_value  # Accumulate MS-SSIM
            rmse_sum += rmse_value  # Accumulate RMSE
            count += 1

            logger.info('{}: PSNR = {:.4f}, SSIM = {:.4f}, LPIPS = {:.4f}, MS-SSIM = {:.4f}, RMSE = {:.4f}'.format(fn, psnr, ssim, lpips_value, ms_ssim_value, rmse_value))

    if opt['test']['use_hr']:
        p /= count
        s /= count
        lpips_avg = lpips_sum / count
        ms_ssim_avg = ms_ssim_sum / count
        rmse_avg = rmse_sum / count
        final_log = "Avg PSNR: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}, MS-SSIM: {:.4f}, RMSE: {:.4f} for model {}".format(
        p, s, lpips_avg, ms_ssim_avg, rmse_avg, os.path.basename(ckpt_path))
        logger.info(final_log)

    print('Testing finished!')

if __name__ == '__main__':
    main()
