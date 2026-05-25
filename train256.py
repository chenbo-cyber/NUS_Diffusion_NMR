import torch
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import data.data256 as Data
import model.model256 as Model
from data.data256 import dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_256_256_condition256.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'], 
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            # train_set = Data.create_dataset(dataset_opt, phase)
            train_set = dataset.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            # val_set = Data.create_dataset(dataset_opt, phase)
            val_set = dataset.create_dataset(dataset_opt, 'valid')
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_mse = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()

                        bsz = visuals['HR'].shape[0]
                        for i in range(bsz):
                            sr_img = torch.complex(visuals['SR'][-bsz+i, 0], visuals['SR'][-bsz+i, 1]).squeeze().float().cpu().numpy()
                            sr_spec = torch.complex(visuals['SR_Spec'][-bsz + i, 0],
                                                    visuals['SR_Spec'][-bsz + i, 1]).squeeze().float().cpu().numpy()
                            hr_img = torch.complex(visuals['HR'][i, 0], visuals['HR'][i, 1]).squeeze().float().cpu().numpy()
                            lr_img = visuals['LR'][i].squeeze().float().cpu().numpy()
                            fake_img = torch.complex(visuals['INF'][i, 0], visuals['INF'][i, 1]).squeeze().float().cpu().numpy()

                            Metrics.save_contour2(
                                nus_spec=fake_img,
                                recon_spec=sr_img,
                                label_spec=hr_img,
                                sr_spec=sr_spec,
                                save_path='{}/{}_{}_hr.png'.format(result_path, current_step, i))

                            # generation
                            eval_mse = Metrics.calculate_mse(sr_img, hr_img)
                            avg_mse += eval_mse
                    avg_mse = avg_mse / (idx*bsz)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_mse))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> mse: {:.4e}'.format(
                        current_epoch, current_step, avg_mse))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_mse, current_step)

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            # sr_img_mode = 'grid'
            # if sr_img_mode == 'single':
            #     # single img series
            #     sr_img = visuals['SR']  # uint8
            #     sample_num = sr_img.shape[0]
            #     for iter in range(0, sample_num):
            #         Metrics.save_img(
            #             Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            # else:
                # grid img
                # sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                # Metrics.save_img(
                #     Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
            
            bsz = visuals['HR'].shape[0]
            for i in range(bsz):
                hr_img = Metrics.tensor2img(visuals['HR'][i])  # uint8
                lr_img = Metrics.tensor2img(visuals['LR'][i])  # uint8
                fake_img = Metrics.tensor2img(visuals['INF'][i])  # uint8
                sr_img = Metrics.tensor2img(visuals['SR'][-bsz+i])

                Metrics.save_img(
                    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, i+1))
                Metrics.save_img(
                    lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, i+1))
                # Metrics.save_img(
                #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, i+1))
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, i+1))

                # generation
                eval_psnr = Metrics.calculate_psnr(sr_img, hr_img)
                eval_ssim = Metrics.calculate_ssim(sr_img, hr_img)

                avg_psnr += eval_psnr
                avg_ssim += eval_ssim

        avg_psnr = avg_psnr / (idx*bsz)
        avg_ssim = avg_ssim / (idx*bsz)

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))