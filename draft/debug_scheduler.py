# debug_scheduler.py
"""
Debug scheduler issue in expert training
"""

import torch
import torch.optim as optim

def test_scheduler_issue():
    """Test to understand why scheduler isn't working properly"""
    
    print("🔍 Debugging scheduler issue...")
    
    # Create a simple model
    model = torch.nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[96, 192, 224], gamma=0.1)
    
    print(f"Initial LR: {optimizer.param_groups[0]['lr']}")
    
    # Test the actual training pattern
    for epoch in range(250):
        # Simulate training step
        optimizer.zero_grad()
        loss = torch.tensor(1.0, requires_grad=True)
        loss.backward()
        optimizer.step()
        
        # Step scheduler AFTER optimizer.step()
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print at key epochs
        if epoch in [0, 95, 96, 97, 191, 192, 193, 223, 224, 225] or epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: LR = {current_lr:.6f}")
    
    print("\n✅ Scheduler debugging complete")

def test_proper_scheduler_usage():
    """Test the correct way to use scheduler in train_expert.py"""
    
    print("\n🔍 Testing proper scheduler usage pattern...")
    
    model = torch.nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[96, 192, 224], gamma=0.1)
    
    # Simulate the exact pattern from train_expert.py
    global_step = 0
    warmup_steps = 15
    base_lr = 0.4
    
    for epoch in range(256):
        # Simulate batches (272 batches per epoch based on our data)
        num_batches = 272
        
        for batch_idx in range(num_batches):
            # Warmup logic (only for first 15 steps total, not per epoch)
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr * lr_scale
            
            # Simulate training step
            optimizer.zero_grad()
            loss = torch.tensor(1.0, requires_grad=True)
            loss.backward()
            optimizer.step()
            global_step += 1
        
        # Scheduler step after each epoch
        if global_step >= warmup_steps:  # Only step scheduler after warmup
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print at key epochs
        if epoch in [0, 1, 2, 95, 96, 97, 191, 192, 193, 223, 224, 225, 255]:
            print(f"Epoch {epoch:3d} (step {global_step:6d}): LR = {current_lr:.6f}")

if __name__ == "__main__":
    test_scheduler_issue()
    test_proper_scheduler_usage()