import torch

def supcon_loss(z_i, z_j, labels, temperature=0.1):
    """
    Supervised Contrastive Learning Loss.
    Pulls together ALL samples with the same label in the batch, creating dense, separated clusters.
    """
    device = z_i.device
    batch_size = labels.shape[0]
    
    # Concatenate the views and duplicate the labels
    z = torch.cat([z_i, z_j], dim=0)
    y = torch.cat([labels, labels], dim=0).to(device)
    
    # Calculate cosine similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    # Remove self-similarity from the diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(mask, -9e15)
    
    # Create a mask of all positive pairs (samples sharing the same label)
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    pos_mask.masked_fill_(mask, 0.0) # Do not count self as positive
    
    # Log-softmax for numerical stability
    max_val, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - max_val.detach()
    log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True))
    
    # Compute the mean log-likelihood for the positive pairs
    pos_counts = pos_mask.sum(dim=1)
    pos_counts[pos_counts == 0] = 1 # Avoid division by zero
    
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_counts
    
    return -mean_log_prob_pos.mean()
