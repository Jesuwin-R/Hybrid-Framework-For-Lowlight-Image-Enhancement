import os
import time
import argparse
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
 
import dataloader
import model
import Myloss
 
 
# ---------------------------------------------------------------------------
# Reproducibility / cudnn perf
# ---------------------------------------------------------------------------
torch.manual_seed(1143)
cudnn.benchmark = True  # fixed input size (image_size x image_size) -> speeds up conv selection
 
 
def format_time(seconds):
    """Formats a duration in seconds as H:MM:SS."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}"
 
 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
 
 
def build_dataloaders(config):
    train_dataset = dataloader.lowlight_loader(
        config.lowlight_images_path,
        image_size=config.image_size,
    )
 
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
 
    return train_loader
 
 
def save_checkpoint(path, epoch, dce_net, optimizer, best_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': dce_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }, path)
 
 
def load_checkpoint_for_resume(path, dce_net, optimizer, device):
    print(f"[Resume] Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    dce_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', -1) + 1
    best_loss = checkpoint.get('best_loss', float('inf'))
    print(f"[Resume] Resuming from epoch {start_epoch}, best_loss so far = {best_loss:.6f}")
    return start_epoch, best_loss
 
 
def find_latest_checkpoint(snapshots_folder):
    latest_path = os.path.join(snapshots_folder, 'latest_checkpoint.pth')
    if os.path.isfile(latest_path):
        return latest_path
    return None
 
 
def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("[WARNING] CUDA GPU not detected — training will run on CPU and be very slow.")
    else:
        print(f"[Info] Using GPU: {torch.cuda.get_device_name(0)}")
 
    os.makedirs(config.snapshots_folder, exist_ok=True)
 
    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    dce_net = model.enhance_net_nopool().to(device)
 
    if config.load_pretrain:
        print(f"[Info] Loading pretrained weights from: {config.pretrain_dir}")
        state = torch.load(config.pretrain_dir, map_location=device)
        # Support both a raw state_dict and a full checkpoint dict.
        if isinstance(state, dict) and 'model_state_dict' in state:
            dce_net.load_state_dict(state['model_state_dict'])
        else:
            dce_net.load_state_dict(state)
    else:
        dce_net.apply(weights_init)
 
    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    train_loader = build_dataloaders(config)
    num_batches_per_epoch = len(train_loader)
    print(f"[Info] Batches per epoch: {num_batches_per_epoch} "
          f"(batch_size={config.train_batch_size}, image_size={config.image_size})")
 
    # -----------------------------------------------------------------
    # Losses (standard Zero-DCE loss terms)
    # -----------------------------------------------------------------
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()
 
    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------
    optimizer = optim.Adam(
        dce_net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
 
    # -----------------------------------------------------------------
    # Resume support
    # -----------------------------------------------------------------
    start_epoch = 0
    best_loss = float('inf')
 
    if config.resume:
        resume_path = config.resume_checkpoint or find_latest_checkpoint(config.snapshots_folder)
        if resume_path and os.path.isfile(resume_path):
            start_epoch, best_loss = load_checkpoint_for_resume(resume_path, dce_net, optimizer, device)
        else:
            print("[Resume] --resume was passed but no checkpoint was found. "
                  "Starting fresh training instead.")
 
    # -----------------------------------------------------------------
    # Mixed precision (disabled — see note below)
    # -----------------------------------------------------------------
    use_amp = config.use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[Info] Automatic Mixed Precision (AMP) enabled.")
    else:
        print("[Info] Automatic Mixed Precision (AMP) DISABLED — training in full fp32 "
              "(this run is deliberately isolating AMP as a variable in debugging color artifacts).")
 
    dce_net.train()
 
    total_epochs = config.num_epochs
    training_start_time = time.time()
 
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        iter_time_accum = 0.0
 
        for iteration, img_lowlight in enumerate(train_loader):
            iter_start_time = time.time()
 
            img_lowlight = img_lowlight.to(device, non_blocking=True)
 
            optimizer.zero_grad(set_to_none=True)
 
            with torch.cuda.amp.autocast(enabled=use_amp):
                enhanced_image_1, enhanced_image, A = dce_net(img_lowlight)
 
                loss_tv = 200 * L_TV(A)
                loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
                loss_col = 5 * torch.mean(L_color(enhanced_image))
                loss_exp = 10 * torch.mean(L_exp(enhanced_image))
 
                loss = loss_tv + loss_spa + loss_col + loss_exp
 
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dce_net.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
 
            running_loss += loss.item()
 
            iter_elapsed = time.time() - iter_start_time
            iter_time_accum += iter_elapsed
 
            if (iteration + 1) % config.display_iter == 0:
                avg_iter_time = iter_time_accum / (iteration + 1)
                iters_remaining_this_epoch = num_batches_per_epoch - (iteration + 1)
                epochs_remaining_after_this = total_epochs - (epoch + 1)
 
                eta_this_epoch = avg_iter_time * iters_remaining_this_epoch
                eta_remaining_epochs = avg_iter_time * num_batches_per_epoch * epochs_remaining_after_this
                eta_total = eta_this_epoch + eta_remaining_epochs
 
                elapsed_total = time.time() - training_start_time
 
                print(
                    f"[Epoch {epoch + 1}/{total_epochs}] "
                    f"[Iter {iteration + 1}/{num_batches_per_epoch}] "
                    f"Loss: {loss.item():.6f} | "
                    f"Elapsed: {format_time(elapsed_total)} | "
                    f"ETA (remaining): {format_time(eta_total)}"
                )
 
        # -------------------------------------------------------------
        # End of epoch bookkeeping
        # -------------------------------------------------------------
        avg_epoch_loss = running_loss / max(1, num_batches_per_epoch)
        epoch_time = time.time() - epoch_start_time
        print(
            f"===> Epoch {epoch + 1}/{total_epochs} complete. "
            f"Avg loss: {avg_epoch_loss:.6f} | "
            f"Epoch time: {format_time(epoch_time)}"
        )
 
        # Always update the "latest" checkpoint so --resume can pick up
        # from exactly where training left off.
        latest_path = os.path.join(config.snapshots_folder, 'latest_checkpoint.pth')
        save_checkpoint(latest_path, epoch, dce_net, optimizer, best_loss)
 
        # Periodic numbered snapshot.
        if (epoch + 1) % config.checkpoint_iter == 0:
            snapshot_path = os.path.join(config.snapshots_folder, f'Epoch{epoch + 1}.pth')
            save_checkpoint(snapshot_path, epoch, dce_net, optimizer, best_loss)
            print(f"[Checkpoint] Saved periodic snapshot: {snapshot_path}")
 
        # Best-model checkpoint (separate from periodic snapshots).
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(config.snapshots_folder, 'best_model.pth')
            save_checkpoint(best_path, epoch, dce_net, optimizer, best_loss)
            print(f"[Best] New best model (avg loss {best_loss:.6f}) saved to: {best_path}")
 
    total_time = time.time() - training_start_time
    print(f"[Done] Training complete. Total time: {format_time(total_time)}")
    print(f"[Done] Best average epoch loss achieved: {best_loss:.6f}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    # Data
    parser.add_argument(
        '--lowlight_images_path', type=str,
        default=r"D:\College\Projects\ICML-Image\dataset\SICE\SICE_ZeroDCE_filtered",
        help="Path to the FILTERED (low-light-only) SICE training images."
    )
    parser.add_argument('--image_size', type=int, default=256,
                         help="Images are resized to (image_size x image_size). "
                              "256 is VRAM-safe for a 6 GB GPU; raise to 384/512 "
                              "only if you lower the batch size accordingly.")
 
    # Optimization
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
 
    # RTX 4050 (6 GB) sizing: batch_size=8 at image_size=256.
    # AMP is now disabled for this run (see --use_amp below), so if you
    # hit a CUDA out-of-memory error, lower this to 4.
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=False,
                         help="Mixed precision training. DISABLED for this run — "
                              "we're isolating AMP as a possible cause of the earlier "
                              "color-artifact issue. Re-enable later once confirmed clean.")
 
    # Logging / checkpointing
    parser.add_argument('--display_iter', type=int, default=10,
                         help="Print progress every N iterations.")
    parser.add_argument('--checkpoint_iter', type=int, default=5,
                         help="Save a periodic numbered snapshot every N epochs.")
    parser.add_argument(
        '--snapshots_folder', type=str,
        default=r"D:\College\Projects\ICML-Image\snapshots_SICE_v2",
        help="NEW folder for this run's checkpoints — kept separate from your "
             "original snapshots_SICE so you can compare old vs. new results."
    )
 
    # Pretrained weights / fine-tuning
    parser.add_argument('--load_pretrain', action='store_true', default=False,
                         help="Initialize from an existing checkpoint (e.g. your LOL-trained model) "
                              "instead of random weights.")
    parser.add_argument('--pretrain_dir', type=str, default='',
                         help="Path to the pretrained .pth file (used only if --load_pretrain is set).")
 
    # Resume support
    parser.add_argument('--resume', action='store_true', default=False,
                         help="Resume training from the latest checkpoint in --snapshots_folder "
                              "(or --resume_checkpoint if given).")
    parser.add_argument('--resume_checkpoint', type=str, default='',
                         help="Explicit checkpoint path to resume from. If omitted, "
                              "'latest_checkpoint.pth' in --snapshots_folder is used.")
 
    config = parser.parse_args()
 
    if config.load_pretrain and not config.pretrain_dir:
        parser.error("--load_pretrain was set but --pretrain_dir was not provided.")
 
    print(f"[Config] Dataset     : {config.lowlight_images_path}")
    print(f"[Config] Snapshots   : {config.snapshots_folder}")
    print(f"[Config] AMP enabled : {config.use_amp}")
    print(f"[Config] Epochs      : {config.num_epochs}")
    print(f"[Config] Batch size  : {config.train_batch_size}")

    train(config)