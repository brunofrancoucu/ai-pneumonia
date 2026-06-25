from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def build_lr_scheduler(optimizer, total_epochs, warmup_epochs, steps_per_epoch):
    """
    Constructs a Warmup + Cosine Decay learning rate scheduler.
    
    Car analogy: 
        -   loss: (supcon contrastive loss), tells how far is left.
        -   Optimizer: (adamW), steering wheel, direction, affects weights.
        -   Scheduler: Gas Pedal.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = (total_epochs - warmup_epochs) * steps_per_epoch
    
    # 1. Warmup: Start at 1% of base_lr and ramp up to 100%
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    
    # 2. Annealing: Decay following a cosine curve down to a near-zero minimum
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    
    # 3. Chain them together
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    return scheduler