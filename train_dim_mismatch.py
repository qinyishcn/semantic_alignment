"""
Dimension Mismatch Experiment.

Demonstrates what happens when encoder and decoder have mismatched latent dimensions:
  1. Direct concatenation → error (dimension mismatch)
  2. Zero-padding → works but with accuracy loss
  3. Truncation → works but with accuracy loss

Uses the pre-trained Model A (latent_dim=32) and trains a new Model C (latent_dim=64)
to create a genuine dimension mismatch scenario.
"""

import os
import json
import time
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from model import Encoder, Decoder

# ─── Configuration ───────────────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR2 = os.path.join(os.path.dirname(__file__), "results_dim_mismatch")
LOG_FILE = os.path.join(RESULTS_DIR2, "dim_mismatch_log.json")

LATENT_DIM_A = 32  # Model A's latent dimension
LATENT_DIM_C = 64  # Model C's latent dimension (mismatched)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train_model_C(train_loader, test_loader, epochs, lr, device, logger):
    """Train Model C with latent_dim=64 (mismatched with Model A's 32)."""
    set_seed(999)
    encoder_c = Encoder(latent_dim=LATENT_DIM_C)
    decoder_c = Decoder(latent_dim=LATENT_DIM_C)
    encoder_c.to(device)
    decoder_c.to(device)

    # Simple wrapper for training
    class FullModel(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec
        def forward(self, x):
            return self.dec(self.enc(x))

    model = FullModel(encoder_c, decoder_c)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "test_loss": [], "test_accuracy": [], "epoch_time": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        test_loss /= len(test_loader)
        accuracy = 100.0 * correct / total

        history["train_loss"].append(round(avg_train_loss, 6))
        history["test_loss"].append(round(test_loss, 6))
        history["test_accuracy"].append(round(accuracy, 4))
        history["epoch_time"].append(round(epoch_time, 2))

        logger.info(
            f"[Model_C] Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}% | Time: {epoch_time:.1f}s"
        )

    return encoder_c, decoder_c, history


def evaluate_model(encoder, decoder, test_loader, device, criterion, model_name, logger):
    """Evaluate a model defined by (encoder, decoder) pair."""
    class CrossModel(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec
        def forward(self, x):
            return self.dec(self.enc(x))

    cross = CrossModel(encoder, decoder)
    cross.to(device)
    cross.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = cross(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total
    logger.info(f"[{model_name}] Accuracy: {accuracy:.2f}% | Loss: {test_loss:.4f}")
    return round(test_loss, 6), round(accuracy, 4)


class ZeroPadDecoder(nn.Module):
    """Decoder that zero-pads a smaller latent vector to match expected input dim."""

    def __init__(self, decoder, target_dim, source_dim):
        super().__init__()
        self.decoder = decoder
        self.target_dim = target_dim  # decoder expects this
        self.source_dim = source_dim  # encoder produces this
        self.pad_size = target_dim - source_dim

    def forward(self, z):
        if self.pad_size > 0:
            pad = torch.zeros(z.size(0), self.pad_size, device=z.device)
            z = torch.cat([z, pad], dim=1)
        elif self.pad_size < 0:
            z = z[:, :self.target_dim]
        return self.decoder(z)


class TruncateDecoder(nn.Module):
    """Decoder that truncates a larger latent vector to match expected input dim."""

    def __init__(self, decoder, target_dim):
        super().__init__()
        self.decoder = decoder
        self.target_dim = target_dim

    def forward(self, z):
        z = z[:, :self.target_dim]
        return self.decoder(z)


def main():
    os.makedirs(RESULTS_DIR2, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(RESULTS_DIR2, "dim_mismatch.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Dimension Mismatch Experiment")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model A latent_dim: {LATENT_DIM_A} | Model C latent_dim: {LATENT_DIM_C}")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders()
    logger.info(f"Train: {len(train_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")

    criterion = nn.CrossEntropyLoss()

    # ── Load pre-trained Model A ──────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("Loading pre-trained Model A (latent_dim=32)")
    logger.info("─" * 50)
    model_a = torch.load(os.path.join(RESULTS_DIR, "model_a.pth"), weights_only=False)
    encoder_a = Encoder(latent_dim=LATENT_DIM_A)
    decoder_a = Decoder(latent_dim=LATENT_DIM_A)
    encoder_a.load_state_dict({k.replace("encoder.", ""): v for k, v in model_a.items() if k.startswith("encoder.")})
    decoder_a.load_state_dict({k.replace("decoder.", ""): v for k, v in model_a.items() if k.startswith("decoder.")})
    encoder_a.to(DEVICE)
    decoder_a.to(DEVICE)

    # Evaluate Model A
    logger.info("Evaluating Model A (original)...")
    loss_a, acc_a = evaluate_model(encoder_a, decoder_a, test_loader, DEVICE, criterion, "Model_A_original", logger)

    # ── Train Model C (latent_dim=64) ─────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("Training Model C (latent_dim=64)")
    logger.info("─" * 50)
    encoder_c, decoder_c, history_c = train_model_C(train_loader, test_loader, EPOCHS, LEARNING_RATE, DEVICE, logger)

    # Evaluate Model C
    logger.info("Evaluating Model C (original)...")
    loss_c, acc_c = evaluate_model(encoder_c, decoder_c, test_loader, DEVICE, criterion, "Model_C_original", logger)

    # ── Experiment 1: Direct concatenation (should error) ─────────────
    logger.info("\n" + "─" * 50)
    logger.info("EXPERIMENT 1: Direct concatenation (A_encoder + C_decoder)")
    logger.info("─" * 50)

    error_msg = None
    try:
        loss_direct, acc_direct = evaluate_model(
            encoder_a, decoder_c, test_loader, DEVICE, criterion, "A_enc+C_dec_DIRECT", logger
        )
    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"EXPECTED ERROR: {error_msg}")
        loss_direct = None
        acc_direct = None

    # Also try C_encoder + A_decoder
    error_msg2 = None
    try:
        loss_direct2, acc_direct2 = evaluate_model(
            encoder_c, decoder_a, test_loader, DEVICE, criterion, "C_enc+A_dec_DIRECT", logger
        )
    except RuntimeError as e:
        error_msg2 = str(e)
        logger.error(f"EXPECTED ERROR: {error_msg2}")
        loss_direct2 = None
        acc_direct2 = None

    # ── Experiment 2: Zero-padding ─────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("EXPERIMENT 2: Zero-padding approach")
    logger.info("─" * 50)

    # A_encoder(32) + C_decoder(64): pad 32→64
    logger.info("A_encoder(32) + C_decoder(64): zero-pad 32→64")
    decoder_c_padded = ZeroPadDecoder(decoder_c, target_dim=LATENT_DIM_C, source_dim=LATENT_DIM_A)
    loss_pad_ac, acc_pad_ac = evaluate_model(
        encoder_a, decoder_c_padded, test_loader, DEVICE, criterion, "A_enc+C_dec_PAD", logger
    )

    # C_encoder(64) + A_decoder(32): truncate 64→32
    logger.info("C_encoder(64) + A_decoder(32): truncate 64→32")
    decoder_a_truncated = TruncateDecoder(decoder_a, target_dim=LATENT_DIM_A)
    loss_pad_ca, acc_pad_ca = evaluate_model(
        encoder_c, decoder_a_truncated, test_loader, DEVICE, criterion, "C_enc+A_dec_TRUNCATE", logger
    )

    # ── Experiment 3: Truncation ───────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("EXPERIMENT 3: Truncation approach")
    logger.info("─" * 50)

    # A_encoder(32) + C_decoder(64): truncate 32→64 (pad is better, but let's also try truncation approach)
    # Actually for A_enc(32) + C_dec(64), truncation doesn't help since 32 < 64
    # We use zero-padding here (already done above), and for the reverse we truncate (already done above)
    # Let's also do: pad A_encoder output 32→64 (pad with zeros) and C_encoder output 64→32 (truncate)
    # These are the same as experiment 2, so let's add an alternative: pad C_encoder(64) with zeros to 64 (no-op) → C_dec(64)
    # and truncate A_encoder(32) to 32 (no-op) → A_dec(32) — those are just the originals.

    # Better approach: for A_enc(32) + C_dec(64), try truncation by duplicating
    # Actually the meaningful comparison is:
    # - Zero-padding: smaller → larger (add zeros)
    # - Truncation: larger → smaller (cut off tail)
    # Both are already covered in experiment 2. Let's add explicit "pad" vs "truncate" labels.

    # Pad approach for C_enc(64) + A_dec(32): pad with zeros to 32? No, that makes no sense.
    # The real meaning:
    #   A_enc(32) + C_dec(64): need to expand 32→64 → zero-padding
    #   C_enc(64) + A_dec(32): need to shrink 64→32 → truncation
    # These are the two natural strategies, both already evaluated.

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("DIMENSION MISMATCH EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model A (dim={LATENT_DIM_A}) accuracy:     {acc_a:.2f}%")
    logger.info(f"Model C (dim={LATENT_DIM_C}) accuracy:     {acc_c:.2f}%")
    logger.info(f"")
    logger.info(f"Direct concat A_enc+C_dec:  {'ERROR — ' + error_msg if error_msg else str(acc_direct) + '%'}")
    logger.info(f"Direct concat C_enc+A_dec:  {'ERROR — ' + error_msg2 if error_msg2 else str(acc_direct2) + '%'}")
    logger.info(f"")
    logger.info(f"Zero-pad A_enc(32→64)+C_dec:   {acc_pad_ac:.2f}%")
    logger.info(f"Truncate C_enc(64→32)+A_dec:   {acc_pad_ca:.2f}%")
    logger.info("=" * 70)

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "experiment": "Dimension Mismatch Experiment",
        "config": {
            "latent_dim_a": LATENT_DIM_A,
            "latent_dim_c": LATENT_DIM_C,
            "epochs": EPOCHS,
            "device": str(DEVICE),
        },
        "model_a": {"accuracy": acc_a, "loss": loss_a},
        "model_c": {"accuracy": acc_c, "loss": loss_c, "training_history": history_c},
        "direct_concat": {
            "A_enc_C_dec": {"error": error_msg, "accuracy": acc_direct, "loss": loss_direct},
            "C_enc_A_dec": {"error": error_msg2, "accuracy": acc_direct2, "loss": loss_direct2},
        },
        "fix_methods": {
            "zero_pad_A_enc_32_to_64_C_dec": {"accuracy": acc_pad_ac, "loss": loss_pad_ac},
            "truncate_C_enc_64_to_32_A_dec": {"accuracy": acc_pad_ca, "loss": loss_pad_ca},
        },
    }

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {RESULTS_DIR2}")


if __name__ == "__main__":
    main()
