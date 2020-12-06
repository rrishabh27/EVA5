import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
# from models import *
# from models_new import *
import os, sys
sys.path.append('..')
import pytorch_ssim
from mega_model import *
# from utils import load_depth_gts

from utils.datasets import *
from utils.utils import *

from utils.google_utils import *
from utils.layers import *
from utils.parse_config import *
from pathlib import Path

# from encoder_midas import EncoderMidas

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    # print('Apex recommended for mixed precision and faster training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'depth_lr': 0.0005, # learning rate when training the depth network ( we use small lr for frozen networks)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])

import os, sys
import numpy as np
import torch
import cv2
from PIL import Image

def load_depth_gts(paths, loc, img_size=512):
    # imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)
    gts = []
    # img_names = glob.glob(os.path.join(input_path, "*"))
    for path in paths:
        img_name = str.split(str.split(path,'images/')[1],'.jpg')[0] + '.png'
        # print('img_name', img_name)
        prefix = [os.path.join(loc,path) for path in os.listdir(os.path.join(loc))
                     if path in img_name]
        # print('prefix', prefix)
        img = Image.open(prefix[0])
        # print(img_size)
        
        # from midas utils
        img = np.array(img)/255
        img = cv2.resize(img,  (int(img_size[-1]),int(img_size[-1])), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(np.expand_dims(img, axis=0)).contiguous().float()
        # return(img)
        
        gts.append(img)

    return torch.cat(gts)


def train():
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)
    flag_depth = opt.depth
    flag_yolo = opt.yolo

    # Image Sizes
    gs = 64  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for f in glob.glob('*_batch*.png') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    # model = Darknet(cfg).to(device)

    # NEWLY ADDED
    model = Mynet(classes=4).to(device) # change classes = 80 for coco dataset

    # for k, v in model.named_parameters(): # name, params
    #     if not (k.startswith('conv_a') or k.startswith('conv_b') or k.startswith('conv_c') or k.startswith('conv1') or k.startswith('conv2') or k.startswith('conv3') or k.startswith('conv4')):
    #         v.requires_grad = False

    ################ OPTIMIZER ##################
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        
        if flag_yolo and not flag_depth: # when training for yolo only
            # if not (k.startswith('conv_a') or k.startswith('conv_b') or k.startswith('conv_c') or k.startswith('conv1') or k.startswith('conv2') or k.startswith('conv3') or k.startswith('conv4') or k.startswith('conv_1_1_up') or k.startswith('conv_2_1_up')):
            #     v.requires_grad = False # freeze the encoder and original yolo layers and only train the custom yolo layers
            if not (k.startswith('conv')):
                v.requires_grad = False # train the yolo layers and freeze the resnext-101 encoder
            else:
                pg0 += [v]
            
        
        elif flag_depth and not flag_yolo: # when training for depth only
            if k.startswith('conv'):
                v.requires_grad = False # freeze the yolo layers and train for depth until we get low loss for depth images
                
            else:
                pg1 += [v] # for using the low lr while training the frozen depth layers

        elif flag_depth and flag_yolo: # when training the whole network
            # if not (k.startswith('conv_a') or k.startswith('conv_b') or k.startswith('conv_c') or k.startswith('conv1') or k.startswith('conv2') or k.startswith('conv3') or k.startswith('conv4') or k.startswith('conv_1_1_up') or k.startswith('conv_2_1_up')):
            #     pg1 += [v] # train the frozen layers with low lr and the custom yolo layers with normal lr (0.01)
            if not (k.startswith('conv')):
                pg1 += [v]
            else:
                pg0 += [v]
        

        # if '.bias' in k:
        #     pg2 += [v]  # biases
        # elif 'Conv2d.weight' in k:
        #     pg1 += [v]  # apply weight_decay
        # else:
        #     pg0 += [v]  # all else
    # or k.startswith('conv_1_1_up') or k.startswith('conv_2_1_up')):

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
    #     optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # optimizer.add_param_group({'params': pg1, 'lr': hyp['depth_lr'], 'weight_decay': hyp['weight_decay'] })
    # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        if flag_yolo and not flag_depth:
            optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        elif flag_depth and not flag_yolo:
            optimizer = optim.SGD(pg1, lr=hyp['depth_lr'], momentum=hyp['momentum'], nesterov=True)
        elif flag_yolo and flag_depth:
            optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
            optimizer.add_param_group({'params': pg1, 'lr': hyp['depth_lr'], 'weight_decay': hyp['weight_decay'] })

    del pg0, pg1, pg2


    ############### WEIGHTS LOADING ###############

    start_epoch = 0
    best_fitness = 0.0
    # attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
    #     # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)
        model.load_state_dict(chkpt['model'], strict=False)

    #     # load model
    #     try:
    #         chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    #         model.load_state_dict(chkpt['model'], strict=False)
    #     except KeyError as e:
    #         s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
    #             "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
    #         raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        # start_epoch = chkpt['epoch'] + 1
        
        del chkpt

    # elif len(weights) > 0:  # darknet format
    #     # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
    #     load_darknet_weights(model, weights)

    # NEWLY ADDED
    print('trainable params: ')
    for name, params in model.named_parameters():
        if params.requires_grad == True:
            print(name)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine https://arxiv.org/pdf/1812.01187.pdf

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, [round(epochs * x) for x in [0.8, 0.9]], 0.1, start_epoch - 1)

    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 10) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', '  depth_loss', '  total_loss', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # if i == 0:
            #     print('PATHS:')
            #     print(paths)
            #     print('imgs size:', imgs.size())
                

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn * 2:
                model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                if ni == n_burn:  # burnin complete
                    print_model_biases(model)

                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])

            # Multi-Scale training
            if opt.multi_scale:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Run model
            depth, pred = model(imgs)

            # print('depth size:')
            # print(depth.size()) # torch.Size([64, 128, 128]) for img input 128 and bs 64

            # print('pred size:')
            # print(pred[0].size()) # torch.Size([64, 128, 128]) for img input 128 and bs 64

            depth_gt_dir = '/content/gdrive/MyDrive/MiDaS/output' # ppe dataset depth
            # depth_gt_dir = '/content/gdrive/MyDrive/S15_NEW/MiDaS/output' # smalcoco depth
            
            # Compute depth loss
            mse_loss = nn.MSELoss()
            ssim = pytorch_ssim.SSIM()
            
            depth_loss = torch.tensor(0).unsqueeze(dim=0).to(device) # just in case opt.depth = 0
            if flag_depth:
                depth_gt = load_depth_gts(paths=paths, loc=depth_gt_dir, img_size = opt.img_size).to(device) # TODO

                # depth_mse_loss = mse_loss(depth.unsqueeze(0).permute(1, 0, 2, 3),
                                    #   depth_gt.unsqueeze(0).permute(1, 0, 2, 3))
                depth_ssim_loss = 1 - ssim(depth.unsqueeze(0).permute(1, 0, 2, 3),
                                            depth_gt.unsqueeze(0).permute(1, 0, 2, 3))
                depth_loss = depth_ssim_loss.unsqueeze(dim=0)
                # depth_loss = (depth_mse_loss + depth_ssim_loss).unsqueeze(dim=0)
                
           
            # Compute yolo loss
            yolo_loss_items = torch.zeros(4).to(device)
            yolo_loss = yolo_loss_items[-1]

            if flag_yolo:
                yolo_loss, yolo_loss_items = compute_loss(pred, targets, model)

            # total loss
            # print('dl', depth_loss, 'yl', yolo_loss)
            total_loss = opt.lambda_depth * depth_loss + opt.lambda_yolo * yolo_loss
            if not flag_depth: # if only training yolo layers then original yolo loss should be backpropagated
                total_loss = yolo_loss

            # print('depth_loss', depth_loss, 'total_loss', total_loss)
                        
            # if not torch.isfinite(total_loss):
            #     print('WARNING: non-finite loss, ending training ', yolo_loss_items)
            #     return results

            # Scale loss by nominal batch_size of 64
            # loss *= batch_size / 64
            
            # NEWLY ADDED
            # train_loss.append(loss)

            # Compute gradient
            # if mixed_precision:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()
            total_loss.backward()

            # Optimize accumulated gradient
            # if ni % accumulate == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     ema.update(model)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(model)

            # Print batch results
            mloss = (mloss * i + yolo_loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 8) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, depth_loss, total_loss, len(targets), img_size)
            pbar.set_description(s)

            # Plot images with bounding boxes
            if ni < 1:
                f = 'train_batch%g.png' % i  # filename
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer:
                    tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')
                    tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      img_size=imgsz_test,
                                      model=ema.ema,
                                      save_json=final_epoch and is_coco,
                                      single_cls=opt.single_cls,
                                      dataloader=testloader)

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Write Tensorboard results
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss', 'depth_loss', 'total_loss']
            # new_list = []
            # new_list = new_list.append(depth_loss.tolist())
            # new_list = new_list.append(total_loss.tolist())
            # print('depth_loss', depth_loss, 'total_loss', total_loss)
            # print('new_list', new_list)
            # print(list(mloss[:-1]), list(results).extend(depth_loss.tolist()).extend(total_loss.tolist()) )
            for x, tag in zip(list(mloss[:-1]) + list(results) + depth_loss.tolist() + total_loss.tolist() , tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save training results
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                         'optimizer': optimizer.state_dict()}
                        #  'optimizer': None if final_epoch else optimizer.state_dict()

            # Save last checkpoint
            torch.save(chkpt, last)

            # Save best checkpoint
            if (best_fitness == fi) and not final_epoch:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                print('saving model')
                torch.save(chkpt, wdir + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    if not opt.evolve:
        plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results
    
    # NEWLY ADDED
    # return loss, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[512], help='[min_train, max-train, test] img sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    # parser.add_argument('--train_new', type=int, default=1, help='train only newly added layers')
    parser.add_argument('--lambda_depth', type=float, default=5.0, help='weight given to the depth loss')
    parser.add_argument('--lambda_yolo', type=float, default=0.2, help='weight given to yolo loss')
    parser.add_argument('--depth', type=int, default=0, help='include the MiDaS-depth training')
    parser.add_argument('--yolo', type=int, default=1, help='include the yolo training')

    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.
    
    # train_loss = [] # NEWLY ADDED
    tb_writer = None
    if not opt.evolve:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter()
            print("Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/")
        except:
            pass

        train()  # train normally

    else:  # Evolve hyperparameters (optional)
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                if method == 1:
                    v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            # results = train()
            
            # NEWLY ADDED
            results = train()

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)
            

            # Plot results
            # plot_evolution_results(hyp)
