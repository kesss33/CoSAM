# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
import datetime
import pdb
from torchvision import utils
import logging


from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
from dataset_info import trainsets_order_map, valid_datasets, train_datasets, membank_datasets
from model import PromptPool, MaskDecoderHQ
from helper import show_anns, show_mask, show_box, show_points, compute_iou, compute_boundary_iou







def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--output", default='work_dirs/hq_sam_b', type=str, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_b", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", default='./pretrained_checkpoint/sam_vit_b_01ec64.pth', type=str, 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    
    # my added args
    parser.add_argument('--CLmethod', default='', type=str, help="The CL method to use")
    parser.add_argument('--distill_weight', default=1.0, type=float, help="The weight of distillation loss")
    parser.add_argument('--ewc_weight', default=0.7, type=float, help="The weight of EWC loss")
    parser.add_argument('--key_match_weight', default=1.0, type=float, help="The weight of key matching loss of L2P")
    parser.add_argument('--trainsets_order', default='train_datasets', type=str, help="to test order robustness")
    parser.add_argument('--best_lr', action='store_true', help="best lr of each task for our method")
    parser.add_argument('--use_prompt_mask', action='store_true', help="diversify prompt selectin to prevent modular collapse")
    parser.add_argument('--show_imgs', action='store_true', help="save a grid of imgs of each dataset for demo")
    parser.add_argument('--demo_results', action='store_true', help="demo compare methods")

    # my added args end

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=24, type=int) # 12
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=11, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(net, train_datasets, valid_datasets, args, task_idx=None, prompt_pool=None):

    # misc.init_distributed_mode(args) # move this line out of main function
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')
    logging.info("Task Index: " + str(task_idx) + '\n')

    seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    debug_dataset_size = None

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train", dataset_size=debug_dataset_size)
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid", dataset_size=debug_dataset_size)
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False,
                                                          eval_ori_resolution=False if args.show_imgs else True)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    
    
    

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        
 
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        BEST_LR = [1e-2, 1e-2, 3e-3, 3e-2, 3e-2, 3e-2, 3e-2, 3e-2]
        if 'l2p' in args.CLmethod:
            if args.best_lr:
                optimizer = optim.Adam(prompt_pool.module.parameters(), lr=BEST_LR[task_idx], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            else:
                optimizer = optim.Adam(prompt_pool.module.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            if args.best_lr:
                optimizer = optim.Adam(net.module.parameters(), lr=BEST_LR[task_idx], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            else:
                optimizer = optim.Adam(net.module.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler, args.CLmethod, train_im_gt_list[0]['dataset_name'], task_idx, prompt_pool=prompt_pool)
    else:
        if 'l2p' in args.CLmethod:
            prompt_pool.module.load_state_dict(torch.load(args.restore_model))
        elif args.restore_model:
            print("restore model from:", args.restore_model)
            net.module.load_state_dict(torch.load(args.restore_model))
        
            
    
        evaluate(args, net, sam, valid_dataloaders, args.visualize, prompt_pool=prompt_pool)


def train(args, net, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler, CLmethod=False, dataset_name='', task_idx=None, prompt_pool=None):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    # copy the net as old_net for CL methods
    if CLmethod in ['lwf', 'ewc']:
        net.eval()
        # old_net = net.module
        # old_net.eval()


        # deep copy net
        print('init a new net to contain old net')
        old_net = MaskDecoderHQ(args.model_type)
        # load state dict with parameter name "module"
        state_dict = {k.replace('module.', ''): v for k, v in net.state_dict().items()}
        old_net.load_state_dict(state_dict)
        # move the old_net to the device of net
        old_net.to(device=args.device)
        old_net.eval()
        print('copying to old_net finished')
        


    net.train()
    _ = net.to(device=args.device)
    
    

    
    for epoch in range(epoch_start,epoch_num): 
        print("*" * 20, "epoch:",epoch, "starts at", datetime.datetime.now().strftime("%H:%M:%S"), "*" * 20)
        print("learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for i, data in enumerate(metric_logger.log_every(train_dataloaders,1000)):
            inputs, labels = data['image'], data['label'] # shape
            # print('inputs.shape, labels.shape', inputs.shape, labels.shape) # [8, 3, 1024, 1024] [8, 1, 1024, 1024]
            

            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # input prompt
            input_keys = ['box','point','noise_mask']
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            try:
                labels_points = misc.masks_sample_points(labels[:,0,:,:])
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                # pdb.set_trace()
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False) # batched_output中每个图片都返回一个字典，包括encoder_embed, img_pe等等；interm_embeds shape: [4,64,64,768]
                # print('interm_embeddings.shape:', interm_embeddings[0].shape) # [4, 64, 64, 768] 4就是batch size
                # print('len of batch:', len(interm_embeddings)) # 4
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            # print('encoder_embedding.shape:', encoder_embedding.shape) # 4,256,64,64
            # print('encoder_embedding.mean,std,max:', encoder_embedding.mean(), encoder_embedding.std(), encoder_embedding.max())
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

            loss_dict = dict()
            loss = 0
            if 'l2p' in args.CLmethod:
                # get the prompt from prompt pool
                if args.use_prompt_mask:
                    start = task_idx * prompt_pool.module.top_k
                    end = (task_idx + 1) * prompt_pool.module.top_k
                    single_prompt_mask = torch.arange(start, end).cuda()
                    prompt_mask = single_prompt_mask.unsqueeze(0).expand(encoder_embedding.shape[0], -1)
                else:
                    prompt_mask = None
                prompt_from_pool, key_similarity, prompt_idx = prompt_pool(x_embed=encoder_embedding, prompt_mask=prompt_mask) # prompt_from_pool shape: [4,25,256]
                # print(prompt_idx[0])
                
                loss_dict.update({"key_similarity": key_similarity, 'prompt_idx0': prompt_idx[0][0], 'prompt_idx1': prompt_idx[0][1], 
                                  'prompt_idx2': prompt_idx[0][2], 'prompt_idx3': prompt_idx[0][3], 'prompt_idx4': prompt_idx[0][4]})
                loss += - args.key_match_weight * key_similarity
                # if args.rank == 0:
                #     wandb.log(loss_dict)
                # losses_reduced_scaled = sum(loss_dict.values())
                # loss_value = losses_reduced_scaled.item()
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # metric_logger.update(training_loss=loss_value, **loss_dict)
                # continue

                # cat prompt_from_pool and sparse_embeddings
                sparse_embeddings = [torch.cat([sparse_embeddings[i_l], prompt_from_pool[i_l].unsqueeze(0)], dim=1) for i_l in range(batch_len)]

            # net is the HQ decoder
            masks_hq = net(
                image_embeddings=encoder_embedding, 
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings, # list,shape(1,2,256) or (1,27,256), depends on whether append prompts from pool
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )
            

            loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            # print(f'masks_hq mean: {masks_hq.mean()}; max: {masks_hq.max()};') # mean around -20 from start to end；max around 20. 
            # print(masks_hq) around -20，
            # print(masks_hq.shape) (4,1,256,256)
            # print(labels.shape, (labels/255).sum()) # (4,1,1024,1024) several thousands to hundred of thousands
            loss += loss_mask + loss_dice 
            
            loss_dict.update({"loss_mask": loss_mask, "loss_dice":loss_dice})
            # print(task_idx)
            if CLmethod=='lwf' and task_idx:
                masks_hq_old = old_net(
                    image_embeddings=encoder_embedding,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    hq_token_only=True,
                    interm_embeddings=interm_embeddings,
                )
                # get the loss between new and old masks
                print('(masks_hq - masks_hq_old).sum():', (masks_hq - masks_hq_old).sum()) # all zeros
                loss_mask_old, loss_dice_old = loss_masks(masks_hq, masks_hq_old, len(masks_hq), soft_labels=True)
                loss += args.distill_weight * (loss_mask_old + loss_dice_old)
                loss_dict["loss_mask_old"] = loss_mask_old
                loss_dict["loss_dice_old"] = loss_dice_old

            if CLmethod=='ewc' and task_idx:
                # load fisher information matrix
                print('load fisher.pth for ewc method')
                fisher = torch.load(args.output + f'/fisher_{args.ewc_weight}.pth')
                ewc_loss = 0
                for (name, param), (_, old_param), (_, imp) in zip(net.named_parameters(), old_net.named_parameters(), fisher.items()):
                    imp = imp.to(param.device)
                    # print(param - old_param)
                    # print('imp.shape, (param - old_param).pow(2).shape:', imp.shape, (param - old_param).pow(2).shape)
                    # print(imp)
                    ewc_loss += torch.sum(imp * (param - old_param).pow(2))
                    # if args.rank == 0:
                    #     wandb.log({'imp_'+name: imp.mean().item(), 'diff_'+name: (param - old_param).pow(2).mean().item(), 'ewc_loss_'+name: ewc_loss.item()})
                loss += args.ewc_weight * ewc_loss
                loss_dict["loss_ewc_weighted"] = args.ewc_weight * ewc_loss




            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            if args.rank == 0:
                wandb.log(loss_dict_reduced)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)


        print("Finished epoch:      ", epoch)
        logging.info("Finished epoch:      " + str(epoch) + '\n')
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(args, net, sam, valid_dataloaders, prompt_pool=prompt_pool if 'l2p' in args.CLmethod else None)
        train_stats.update(test_stats)
        
        net.train()  

        # if epoch % args.model_save_fre == 0: change to save first and last model
        if epoch % (args.max_epoch_num-1) == 0:
            model_name = '/'+args.CLmethod + '_'+dataset_name+"_epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            if 'l2p' in args.CLmethod:
                misc.save_on_master(prompt_pool.module.state_dict(), args.output + model_name)
            else:
                misc.save_on_master(net.module.state_dict(), args.output + model_name) # args.output="work_dirs/hq_sam_b"
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")

    if args.CLmethod == 'ewc':
        print('Calculate Fisher Information Matrix')
        # calculate fisher information matrix
        fisher = {}
        for n,p in net.named_parameters():
            fisher[n] = torch.zeros_like(p)
        net.eval()
        for i, data in enumerate(train_dataloaders):
            inputs, labels = data['image'], data['label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            input_keys = ['box','point','noise_mask']
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            try:
                labels_points = misc.masks_sample_points(labels[:,0,:,:])
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

            masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )

            loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            # print(labels.shape, (labels/255).sum()) # (4,1,1024,1024) several thousands to hundred of thousands
            loss = loss_mask + loss_dice
            optimizer.zero_grad()
            loss.backward()

            for name, param in net.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

        for name in fisher:
            fisher[name] /= len(train_dataloaders)
        # sync the fisher of all gpus and save
        # fisher_reduced = misc.reduce_dict(fisher)
        if misc.is_main_process():
            torch.save(fisher, args.output + f'/fisher_{args.ewc_weight}.pth')
            print('fisher saved at', args.output + f'/fisher_{args.ewc_weight}.pth')
            





def evaluate(args, net, sam, valid_dataloaders, visualize=False, prompt_pool=None):
    net.eval()
    print("Validating...")
    test_stats = {}
    if args.show_imgs:
        img_ls = []

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))
        iou_equal_biou = 0
        for i, data_val in enumerate(metric_logger.log_every(valid_dataloader,1000)):

            if args.show_imgs:
                inputs_val, labels_val = data_val['image'], data_val['label']
                labels_val = labels_val.repeat(1,3,1,1)
                img_ls.append(inputs_val)
                img_ls.append(labels_val)
                if k in [0, 2, 5, 6, 7]:
                    inputs_val = torch.cat(img_ls, dim=0)
                    img_grid = utils.make_grid(inputs_val, nrow=args.batch_size_valid, normalize=True)
                    utils.save_image(img_grid, f'img_grid_{k}_vv2.png')
                    img_ls = []
                break
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                    # dict_input['boxes'] = None
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            
            if prompt_pool is not None:
                # get the prompt from prompt pool
                prompt_from_pool, loss_key_match, prompt_idx = prompt_pool(x_embed=encoder_embedding) # prompt_from_pool shape: [4,25,256]
                sparse_embeddings = [torch.cat([sparse_embeddings[i_l], prompt_from_pool[i_l].unsqueeze(0)], dim=1) for i_l in range(batch_len)]
                
            masks_sam, masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
            )

            if args.CLmethod == 'sam':
                masks_hq = torch.cat([batched_output[i_l]['masks'].to(torch.float64) for i_l in range(batch_len)], dim=0)


            if args.demo_results and i < 29: # 各task中最小的validset就是29
                utils.save_image(inputs_val, f'demo_imgs/task{k}_id{i}_img.png', normalize=True)
                utils.save_image(labels_val, f'demo_imgs/task{k}_id{i}_gt.png')
                utils.save_image(masks_hq, f'demo_imgs/task{k}_id{i}_{args.CLmethod}.png')
                

            # print('masks_hq, labels_ori:', '*' * 40)
            # print(masks_hq.shape, (masks_hq>0).sum()) # zero positive point
            # print(labels_ori.shape, (labels_ori>127).sum()) # zero >127 point

            iou = compute_iou(masks_hq,labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq,labels_ori)
            if iou == boundary_iou:
                iou_equal_biou += 1
            # if i == 0:
            #     print('masks_hq,labels_ori, iou, boundary_iou:', '*' * 40)
            #     print(masks_hq,labels_ori, iou, boundary_iou)
            # exit()

            # save input_val, labels_val, masks_hq
            
                
                

            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(masks_hq.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    print('base:', base)
                    save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)
                    
            loss_dict = {"iou_"+str(k)+valid_datasets[k]['name']: iou, "boundary_iou_"+str(k)+valid_datasets[k]['name']: boundary_iou}
            if prompt_pool is not None:
                loss_dict.update({'prompt_idx0': prompt_idx[0][0], 'prompt_idx1': prompt_idx[0][1], 
                                  'prompt_idx2': prompt_idx[0][2], 'prompt_idx3': prompt_idx[0][3], 'prompt_idx4': prompt_idx[0][4]})

            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)
        if args.show_imgs:
            continue

        print('============================')
        print('iou_equal_biou_ratio:', iou_equal_biou/len(valid_dataloader))
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        logging.info(metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        # add one item as the average of all the stat
        avg = sum(resstat.values()) / len(resstat.values())
        resstat['avg'] = avg
        # if args.rank == 0:
        #     wandb.log(resstat)
        test_stats.update(resstat)
    
    if args.show_imgs or args.demo_results:
        exit()

    return test_stats


if __name__ == "__main__":

    args = get_args_parser()
    misc.init_distributed_mode(args) # this line is moved from main function


    import wandb
    # os.environ['WANDB_API_KEY'] = 'KEY'
    # os.environ['WANDB_MODE'] = 'offline'
    
    if args.rank == 0:
        # wandb.init(project="SAM-HQ Continual Adaptation", entity="zzsyjl")
        wandb.init(mode='disabled')



    logging.basicConfig(filename=f'experiment_result_{args.CLmethod}_epoch_{args.max_epoch_num}_lr_{args.learning_rate}.log', 
                        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')



    net = MaskDecoderHQ(args.model_type)
    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    net.module.load_state_dict(torch.load('./pretrained_checkpoint/pretrain_maskDecoderHQ.pth'))
    



    if 'l2p' in args.CLmethod:
        prompt_pool = PromptPool()
        prompt_pool.cuda()
        prompt_pool = torch.nn.parallel.DistributedDataParallel(prompt_pool, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        for task_idx, trainset in enumerate(train_datasets):
            main(net, [trainset], valid_datasets, args, task_idx, prompt_pool=prompt_pool)
    elif args.CLmethod == 'er':
        for task_idx, trainset in enumerate(train_datasets):
            main(net, [trainset]+membank_datasets[:task_idx], valid_datasets, args, task_idx)
    else: # other cl methods
        for task_idx, trainset in enumerate(train_datasets):
            main(net, [trainset], valid_datasets, args, task_idx)