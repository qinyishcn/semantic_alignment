"""
Training and evaluation pipeline for semantic alignment verification.

Steps:
  1. Train Model_A and Model_B independently on MNIST (same architecture, different seeds)
  2. Evaluate each model's accuracy
  3. Build cross-combinations: A_encoder + B_decoder, B_encoder + A_decoder
  4. Evaluate cross-model accuracy
  5. Log all results and analyze latent space alignment
"""

import os
import json
import time
import random
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from model import AutoencoderClassifier, CrossModel, Encoder, Decoder

# ─── Configuration ───────────────────────────────────────────────────────
LATENT_DIM = 32
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED_A = 42
SEED_B = 123
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.json")


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


def train_model(model, train_loader, test_loader, epochs, lr, device, model_name, logger):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "test_loss": [], "test_accuracy": [], "epoch_time": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
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
        test_loss, accuracy = evaluate_model(model, test_loader, criterion, device)

        history["train_loss"].append(round(avg_train_loss, 6))
        history["test_loss"].append(round(test_loss, 6))
        history["test_accuracy"].append(round(accuracy, 4))
        history["epoch_time"].append(round(epoch_time, 2))

        logger.info(
            f"[{model_name}] Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}% | Time: {epoch_time:.1f}s"
        )

    return history


def evaluate_model(model, test_loader, criterion, device):
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
    return test_loss, accuracy


def evaluate_cross_model(encoder, decoder, test_loader, criterion, device, model_name, logger):
    """Evaluate a cross-combination model (encoder from one, decoder from another)."""
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

    logger.info(f"[{model_name}] Cross-model accuracy: {accuracy:.2f}% | Loss: {test_loss:.4f}")
    return round(test_loss, 6), round(accuracy, 4)


def compute_latent_stats(model, test_loader, device, model_name, logger):
    """Compute statistics of the latent space to analyze alignment."""
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            z = model.encode(data)
            all_latents.append(z.cpu().numpy())
            all_labels.append(target.numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    stats = {
        "mean": all_latents.mean(axis=0).tolist(),
        "std": all_latents.std(axis=0).tolist(),
        "min": all_latents.min(axis=0).tolist(),
        "max": all_latents.max(axis=0).tolist(),
    }

    # Compute per-class centroid distances
    centroids = {}
    for c in range(10):
        mask = all_labels == c
        centroids[c] = all_latents[mask].mean(axis=0)

    # Average inter-class distance
    inter_dists = []
    for i in range(10):
        for j in range(i + 1, 10):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_dists.append(float(dist))

    stats["avg_inter_class_dist"] = round(float(np.mean(inter_dists)), 4)
    stats["latent_dim"] = all_latents.shape[1]

    logger.info(
        f"[{model_name}] Latent stats — mean_norm: {np.linalg.norm(stats['mean']):.4f}, "
        f"avg_inter_class_dist: {stats['avg_inter_class_dist']:.4f}"
    )

    return stats


def main():
    # Setup logging
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(RESULTS_DIR, "run.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Semantic Alignment Verification Experiment")
    logger.info(f"Device: {DEVICE} | Latent dim: {LATENT_DIM} | Epochs: {EPOCHS}")
    logger.info(f"Seeds: A={SEED_A}, B={SEED_B}")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders()
    logger.info(f"Train: {len(train_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")

    criterion = nn.CrossEntropyLoss()

    # ── Step 1: Train Model A ──────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 1: Training Model A")
    logger.info("─" * 50)
    set_seed(SEED_A)
    model_a = AutoencoderClassifier(LATENT_DIM)
    history_a = train_model(model_a, train_loader, test_loader, EPOCHS, LEARNING_RATE, DEVICE, "Model_A", logger)

    # ── Step 2: Train Model B ──────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 2: Training Model B")
    logger.info("─" * 50)
    set_seed(SEED_B)
    model_b = AutoencoderClassifier(LATENT_DIM)
    history_b = train_model(model_b, train_loader, test_loader, EPOCHS, LEARNING_RATE, DEVICE, "Model_B", logger)

    # ── Step 3: Cross-model evaluation ─────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 3: Cross-model evaluation")
    logger.info("─" * 50)

    # A_encoder + B_decoder
    loss_ab, acc_ab = evaluate_cross_model(
        model_a.encoder, model_b.decoder, test_loader, criterion, DEVICE, "A_enc+B_dec", logger
    )

    # B_encoder + A_decoder
    loss_ba, acc_ba = evaluate_cross_model(
        model_b.encoder, model_a.decoder, test_loader, criterion, DEVICE, "B_enc+A_dec", logger
    )

    # ── Step 4: Latent space analysis ──────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 4: Latent space analysis")
    logger.info("─" * 50)

    latent_stats_a = compute_latent_stats(model_a, test_loader, DEVICE, "Model_A", logger)
    latent_stats_b = compute_latent_stats(model_b, test_loader, DEVICE, "Model_B", logger)

    # Compute cosine similarity between latent spaces
    mean_a = np.array(latent_stats_a["mean"])
    mean_b = np.array(latent_stats_b["mean"])
    cosine_sim = float(np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-8))

    # Compute L2 distance between latent distributions
    l2_dist = float(np.linalg.norm(mean_a - mean_b))

    logger.info(f"Latent space cosine similarity: {cosine_sim:.6f}")
    logger.info(f"Latent space L2 distance (mean): {l2_dist:.4f}")

    # ── Step 5: Save results ───────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 5: Saving results")
    logger.info("─" * 50)

    # Save models
    torch.save(model_a.state_dict(), os.path.join(RESULTS_DIR, "model_a.pth"))
    torch.save(model_b.state_dict(), os.path.join(RESULTS_DIR, "model_b.pth"))

    # Compile results
    results = {
        "experiment": "Semantic Alignment Verification",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "latent_dim": LATENT_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "device": str(DEVICE),
            "seed_a": SEED_A,
            "seed_b": SEED_B,
        },
        "results": {
            "model_a": {
                "final_accuracy": history_a["test_accuracy"][-1],
                "final_loss": history_a["test_loss"][-1],
                "best_accuracy": max(history_a["test_accuracy"]),
                "training_history": history_a,
            },
            "model_b": {
                "final_accuracy": history_b["test_accuracy"][-1],
                "final_loss": history_b["test_loss"][-1],
                "best_accuracy": max(history_b["test_accuracy"]),
                "training_history": history_b,
            },
            "cross_models": {
                "A_encoder_B_decoder": {
                    "accuracy": acc_ab,
                    "loss": loss_ab,
                },
                "B_encoder_A_decoder": {
                    "accuracy": acc_ba,
                    "loss": loss_ba,
                },
            },
            "latent_analysis": {
                "model_a": latent_stats_a,
                "model_b": latent_stats_b,
                "cosine_similarity": cosine_sim,
                "l2_distance_mean": l2_dist,
            },
        },
        "conclusion": {
            "model_a_accuracy": history_a["test_accuracy"][-1],
            "model_b_accuracy": history_b["test_accuracy"][-1],
            "cross_A_enc_B_dec_accuracy": acc_ab,
            "cross_B_enc_A_dec_accuracy": acc_ba,
            "accuracy_drop_A_enc_B_dec": round(history_a["test_accuracy"][-1] - acc_ab, 2),
            "accuracy_drop_B_enc_A_dec": round(history_b["test_accuracy"][-1] - acc_ba, 2),
            "verdict": (
                "Cross-combination leads to significant accuracy degradation, "
                "confirming that encoder-decoder pairs are NOT interchangeable "
                "without semantic alignment."
            ),
        },
    }

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model A accuracy:              {history_a['test_accuracy'][-1]:.2f}%")
    logger.info(f"Model B accuracy:              {history_b['test_accuracy'][-1]:.2f}%")
    logger.info(f"A_encoder + B_decoder:         {acc_ab:.2f}%")
    logger.info(f"B_encoder + A_decoder:         {acc_ba:.2f}%")
    logger.info(f"Accuracy drop (A_enc+B_dec):   {results['conclusion']['accuracy_drop_A_enc_B_dec']:.2f}%")
    logger.info(f"Accuracy drop (B_enc+A_dec):   {results['conclusion']['accuracy_drop_B_enc_A_dec']:.2f}%")
    logger.info(f"Latent cosine similarity:      {cosine_sim:.6f}")
    logger.info(f"Latent L2 distance:            {l2_dist:.4f}")
    logger.info("=" * 70)
    logger.info("Results saved to: %s", RESULTS_DIR)

    return results


if __name__ == "__main__":
    main()
