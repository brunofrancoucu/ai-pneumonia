import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from model.utils.dataloader import get_dataloader
from model.utils.loss import supcon_loss
from model.utils.scheduler import build_lr_scheduler
from model.augmentation import augmented_dataloader
from model.encoder import ContrastiveEncoder
from model.classification import LinearEvaluator

def train_resnet(device, epochs):
    """
    Phase 1: Contrastive Pre-Training (Resnet training)
    """
    augmented_images = augmented_dataloader("src/dataset/train") # Dataset
    model = ContrastiveEncoder().to(device) # ResNet50
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # AdamW
    scheduler = build_lr_scheduler(optimizer=optimizer, total_epochs=epochs, warmup_epochs=10, steps_per_epoch=len(augmented_images)) # Cosine lr scheduler

    model.train() # Enable
    for epoch in range(epochs):
        total_loss = 0
        for (view_1, view_2), labels, _ in augmented_images:
            view_1, view_2 = view_1.to(device), view_2.to(device)
            labels = labels.to(device) # Send labels to GPU

            optimizer.zero_grad()
            
            # Forward pass: extract z vectors
            _, z_i = model(view_1)
            _, z_j = model(view_2)
            
            loss = supcon_loss(z_i, z_j, labels) # labeled clusters
            loss.backward()     # Backpropagate params
            optimizer.step()    # Calc loss
            scheduler.step()    # smooth lr curve
            total_loss += loss.item()
            
        checkpoint_path = f'src/weights/__contrastive_encoder_epoch_{epoch+1}.pth'
        if (epoch + 1) % 10 == 0: torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch+1} | Contrastive Loss: {total_loss/len(augmented_images):.4f}")
    
    # Save the trained encoder
    torch.save(model.state_dict(), 'src/weights/__contrastive_encoder.pth')
    print(f"Trained model saved at 'src/weights/__contrastive_encoder.pth'")
    return model

def train_classifier(device, epochs=20,encoder_epoch=10):
    """
    Phase 2: Linear Evaluation (Classification training)
    """
    views = get_dataloader("src/dataset/train", shuffle=True)
    # Load Resnet
    encoder = ContrastiveEncoder().to(device)
    encoder.load_state_dict(torch.load(f'src/weights/__contrastive_encoder_epoch_{encoder_epoch}.pth'))
    
    # 1. Discard projection head & freeze backbone weights
    for param in encoder.parameters():
        param.requires_grad = False # Freeze trained backbone
    encoder.eval() # Ensure BatchNorm statistics are frozen
    
    class_head = LinearEvaluator(feature_dim=encoder.feature_dim).to(device)
    optimizer = torch.optim.AdamW(class_head.parameters(), lr=3e-4) # will optimize the linear head parameters
    criterion = torch.nn.CrossEntropyLoss()

    class_head.train() # Enable
    for epoch in range(epochs):
        total_loss = 0
        for images, labels, _ in views:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Pass through frozen backbone using torch.no_grad() to save memory
            with torch.no_grad():
                h, _ = encoder(images)
            
            # Forward pass through classification head
            logits = class_head(h)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Classification Loss: {total_loss/len(views):.4f}")

    # Save Head
    torch.save(class_head.state_dict(), f"src/weights/classifier_head_epoch_{encoder_epoch}.pth")
    print(f"Saved classifier head to 'src/weights/classifier_head_epoch_{encoder_epoch}.pth'")

def fine_tune(device, epochs, encoder_epoch=10):
    """
    Phase 3: Full Network Fine-Tuning
    Backbone and Classification train together
    """
    views = get_dataloader("src/dataset/train", shuffle=True) 
    criterion = torch.nn.CrossEntropyLoss()
    # Load ResNet
    encoder = ContrastiveEncoder().to(device)
    encoder.load_state_dict(torch.load(f'src/weights/__contrastive_encoder_epoch_{encoder_epoch}.pth'))
    # Load Linear Evaluator
    class_head = LinearEvaluator(feature_dim=encoder.feature_dim).to(device)
    class_head.load_state_dict(torch.load(f'src/weights/classifier_head_epoch_{encoder_epoch}.pth'))
    
    # 1. Unfreeze the backbone
    for param in encoder.parameters():
        param.requires_grad = True
    
    # 2. Use a microscopic learning rate so you don't destroy the latent space
    fine_tune_optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(class_head.parameters()), 
        lr=1e-5 # 10x to 30x smaller than your previous LR
    )
    
    encoder.train() # Enable
    class_head.train()
    for epoch in range(epochs): # 5-10
        total_loss = 0
        for images, labels, _ in views:
            images, labels = images.to(device), labels.to(device)
            
            fine_tune_optimizer.zero_grad()
            
            # Forward pass through the entire unfrozen network
            h, _ = encoder(images)
            logits = class_head(h)
            loss = criterion(logits, labels)
            loss.backward()
            fine_tune_optimizer.step()
            total_loss += loss.item()
            
        print(f"Fine-Tune Epoch {epoch+1} | Loss: {total_loss/len(views):.4f}")
    
    # Save Model
    torch.save(encoder.state_dict(), f'src/weights/finetuned_encoder_epoch_{encoder_epoch}.pth')
    print(f"Saved fine tuned model to src/weights/finetuned_encoder_epoch_{encoder_epoch}.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_resnet(device, epochs=50) # Phase 1
    train_classifier(device, epochs=20, encoder_epoch=10) # Phase 2
    fine_tune(device, epochs=7, encoder_epoch=10) # Phase 3
