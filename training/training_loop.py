# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

#----------------------------------------------------------------------------
#Included for monitoring the network during training
import matplotlib.pyplot as plt
import torch.nn.functional as F

def edm_sampler(
    net, latents, class_labels=None, known_data = None, lbl_lims = None,
    randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, cur_nimg='', TI_val='', dir = None, sigma=None
):
    
    if TI_val.shape[1]>1 and sigma is not None:
        x0 = TI_val[:,0]+ torch.randn_like(TI_val[:,0])*sigma.cpu()
        TI_val = torch.cat([x0[:,None], TI_val[:,1,None]], dim=1)
    
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents * t_steps[0]
    
    fig3, axs3 = plt.subplots(1,1, figsize=(16,4), dpi=200)
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        if net.model.label_head is not None:
            denoised, pred_label = net(x_hat, t_hat, class_labels)
        else:
            denoised = net(x_hat, t_hat, class_labels)
        
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            if net.model.label_head is not None:
                denoised, pred_label = net(x_next, t_next, class_labels)
            else:
                denoised = net(x_next, t_next, class_labels)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        if (x_next.shape[1]==1 and i==0) and net.model.label_head is not None: 
            plotfirst = denoised.detach().cpu() #save for plotting
        
        # if (x_next.shape[1]>1 and i%2==0) and i>2:
        #     axs3.scatter(x_next[:,0].detach().flatten().cpu(),x_next[:,1].detach().flatten().cpu(), alpha=0.1)
        # elif (x_next.shape[1]==1 and i%2==0) and i>2:
        #     axs3.hist(denoised.detach().flatten().cpu(), alpha=0.1, label=i, bins=50, density=True)
    x_next = x_next.detach().cpu()

    if net.model.label_head is not None:
        pred_label = lbl_lims.minlbl+((pred_label.detach().cpu().numpy()+1)/2)*(lbl_lims.maxlbl-lbl_lims.minlbl)
        pred_label = np.round(pred_label,2)

    if net.model.label_head is not None:
        fig1, axs1 = plt.subplots(5,2, sharey=True, figsize=(8,15), dpi=200,sharex=True)
        for i in range(5):
          ax = axs1[i,0].imshow(plotfirst[i,0].detach().cpu(), cmap='jet')
          plt.colorbar(ax)
          ax = axs1[i,1].imshow(x_next[i,0].detach().cpu(), cmap='jet')
          plt.colorbar(ax)
          if net.model.label_head is not None:
              label = f'az:{pred_label[i][0]:.1f}'+' - '
              if len(pred_label[i])>1:
                label+= f'rM:{pred_label[i][1]:.1f}'+' - '
                label+= f'rm:{pred_label[i][2]:.1f}'+' - '
              axs1[i,1].set_title(label)
        fig1.savefig(f'{dir}//f_{cur_nimg//1000:06d}.png')
    
    else:
        fig1, axs1 = plt.subplots(3,2, sharey=True, figsize=(7,5), dpi=200)
        k=0
        for i in range(1):
            for j in range(2):
                ax = axs1[i,j].imshow(x_next[k,0].detach().cpu(), cmap='jet')
                plt.colorbar(ax)
                k+=1
        ax = axs1[-2,0].imshow(x_next.mean(0)[0].detach().cpu(), cmap='jet')
        plt.colorbar(ax)
        ax = axs1[-2,1].imshow(x_next.std(0)[0].detach().cpu(), cmap='jet')
        plt.colorbar(ax)
        ax = axs1[-1,0].imshow(x_next.mean(0)[1].detach().cpu(), cmap='coolwarm')
        plt.colorbar(ax)
        ax = axs1[-1,1].imshow(x_next.std(0)[1].detach().cpu(), cmap='coolwarm')
        plt.colorbar(ax)

        fig1.savefig(f'{dir}//f_{cur_nimg//1000:06d}.png')
        
    if x_next.shape[1]>1:
        if TI_val.shape[1]>1 and sigma is not None: fig1.suptitle(sigma)
        axs3.scatter(TI_val[:,0].flatten(),TI_val[:,1].flatten().cpu(), c='gray', alpha=1)
        axs3.scatter(x_next[:,0].detach().flatten().cpu(),x_next[:,1].detach().flatten().cpu(), c='r', alpha=0.1)
        axs3.set_xlim([-1.6,1.6])
        axs3.set_ylim([-2.5,3.5])
        fig3.savefig(f'{dir}//h3_{cur_nimg//1000:06d}.png')
        plt.close('all')
        
        nn = len(TI_val[:,0].flatten())
        nnn = len(x_next[:,0].detach().cpu().flatten())
        fig1, axs1 = plt.subplots(1,1, figsize=(16,4), dpi=200)
        axs1.hist(TI_val[:,0].flatten(), color='gray', alpha=1, weights=np.ones(nn)/nn)
        axs1.hist(x_next[:,0].detach().cpu().flatten(), color='r', alpha=0.1, weights=np.ones(nnn)/nnn)
        fig1.savefig(f'{dir}//h1_{cur_nimg//1000:06d}.png')

        fig1, axs1 = plt.subplots(1,1, figsize=(16,4), dpi=200)
        axs1.hist(TI_val[:,1].flatten(), color='gray', alpha=1, density=True, bins=50)
        axs1.hist(x_next[:,1].detach().cpu().flatten(), color='r', alpha=0.1, density=True, bins=50)
        fig1.savefig(f'{dir}//h2_{cur_nimg//1000:06d}.png')

    else:
        axs3.hist(denoised.detach().flatten().cpu(), alpha=0.1, label=num_steps, bins=50, density=True)
        axs3.set_xlim([-2,2])
        axs3.legend()
        fig3.savefig(f'{dir}//h_{cur_nimg//1000:06d}.png')

    plt.close('all')
    torch.cuda.empty_cache()

    return None


#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 192000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 5000,      # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu

    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    if loss_kwargs.class_name == 'training.loss.EDMLoss_w': dataset_obj.return_weights = True
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    
    val_dataset = dnnlib.util.construct_class_by_name(n_ti=1, **dataset_kwargs)
    
    
    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    dist.print0(loss_fn.P_mean, loss_fn.P_std)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    # added stuff just in case one wants to implement lr step decay
    size = dataset_obj.image_shape
    size.insert(0,10)
    LR_halfckp = np.array([total_kimg*1000+1, total_kimg*1000+1])
    if cur_nimg>=LR_halfckp[-2]:
        idx_ckp = len(LR_halfckp)-1
    else:
        idx_ckp = np.where(cur_nimg<LR_halfckp)[0].min()
    optimizer_kwargs['lr'] = optimizer_kwargs['lr']/(2**idx_ckp)
    
    
    if ddp.module.model.label_head is not None:
        #rescaled cubic weight
        cubic_w = size[-1]*size[-2]*1/dataset_obj.label_dim  #1 is number of channels, to be changed with more than one parameter
        alpha = .1 # 0.5 means that the "cube" (surface now) has area = 1/2 of the image
        W_loss2 = alpha*cubic_w
        
    if dist.get_rank() == 0:
        if ddp.module.model.label_head is not None:
            TI_val, _ = next(dataset_iterator)
            TI_val = TI_val.detach().clone().cpu()[:10]
            del _
        else:
            pass #see later

    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):

                images, labels = next(dataset_iterator)
                #images = images.to(device).to(torch.float32) / 127.5 - 1 -- No need, geoimages are normalized by the dataloader (also the source is not RGB)
                images = images.to(device)
                labels = labels.to(device)
                
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                
                if ddp.module.model.label_head is not None:
                    training_stats.report('Loss/loss_img', loss[0].detach().cpu())
                    training_stats.report('Loss/loss_lbl', loss[1].detach().cpu())
                    loss = (loss[0].sum() + loss[1].sum()*W_loss2) #
                    loss.mul(loss_scaling / batch_gpu_total).backward()
                else:
                    loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                training_stats.report('Loss/loss', loss.detach().cpu())
                
                del loss, images
                
                total_norm = 0.0
                for p in ddp.module.model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad.detach()).any() or torch.isinf(p.grad.detach()).any(): print('NAN OR INF DETECTED IN GRADS')
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm ** 0.5
                training_stats.report('Loss/norm_grads', total_norm)
                training_stats.report('Loss/lr', (optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)))
                del total_norm                           
                
                
        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        
        optimizer.step()
        torch.distributed.barrier()
        
        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        
        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        if cur_nimg > LR_halfckp[idx_ckp]:
            optimizer_kwargs['lr'] = optimizer_kwargs['lr']/2
            idx_ckp+=1

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.3f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))
        
        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                print('Save network snapshot')
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory
        
        # Validation of EMA
      
          
        # Compute validation, not using loss_fn
        with torch.inference_mode():
            TI_val = torch.stack([torch.from_numpy(val_dataset.getvalidation()[0]) for _ in range(30) ]).to(device)
            sigma = torch.randn(
                TI_val.shape[0], 1, 1, 1,
                device=device)
            sigma = (sigma * loss_fn.P_std + loss_fn.P_mean).exp()
    
            weight = (sigma**2 + loss_fn.sigma_data**2) / (sigma * loss_fn.sigma_data)**2
            noise = torch.randn_like(TI_val) * sigma
            
            pred = ema.eval()(TI_val + noise, sigma, None)
            val_loss = (weight * (pred - TI_val).square()).mean()
    
        training_stats.report("Loss/val_loss", val_loss.cpu())
        
        if dist.get_rank() == 0:  
            #plot stuff at fixed intervals
            if ((snapshot_ticks is not None) and (done or cur_tick % 10 == 0)) or (cur_tick<10):
                edm_sampler(ema, latents= torch.randn(size).to(device).detach(),
                        lbl_lims = dataset_obj if ddp.module.model.label_head is not None else None,
                        cur_nimg = cur_nimg, TI_val=TI_val.detach().cpu(), dir = run_dir, sigma=None)
            
            images, labels = next(dataset_iterator)
        del TI_val, val_loss
        torch.distributed.barrier()
            
        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and dist.get_rank() == 0: #and cur_tick != 0 I want it at 0
            print('Save full dump of training state')
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
        
        torch.cuda.empty_cache()
        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1        
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
        torch.cuda.empty_cache()
          
    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------