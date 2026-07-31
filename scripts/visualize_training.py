"""
visualize_training.py
---------------------
Đọc metrics.json từ output_dir và vẽ biểu đồ train/val loss.

Dùng:
    python scripts/visualize_training.py --metrics outputs/internvl3_2b/<timestamp>/metrics.json
    python scripts/visualize_training.py --metrics outputs/internvl3_2b/<timestamp>/metrics.json --save plot.png
"""

import argparse
import json
import os
import sys


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_metrics(base_dir: str = "./outputs/internvl3_2b") -> str:
    """Tự tìm file metrics.json mới nhất nếu không chỉ định path."""
    from pathlib import Path
    candidates = sorted(
        Path(base_dir).glob("*/metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy metrics.json trong {base_dir}")
    return str(candidates[0])


def plot_metrics(metrics: dict, save_path: str | None = None, title_prefix: str = ""):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[ERROR] matplotlib chưa được cài. Chạy: pip install matplotlib")
        sys.exit(1)

    train_data = metrics.get("train_loss", [])
    val_data   = metrics.get("val_loss", [])
    epoch_data = metrics.get("epoch_summary", [])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title_prefix}Training Metrics", fontsize=14, fontweight="bold")

    # ── Plot 1: Train loss theo step ──────────────────────────────────────────
    ax1 = axes[0]
    if train_data:
        steps  = [d["step"] for d in train_data]
        losses = [d["loss"] for d in train_data]
        epochs_list = sorted(set(d["epoch"] for d in train_data))

        # Tô màu nền theo epoch
        colors = ["#eef4ff", "#fff8ee", "#f0fff4"]
        for idx, ep in enumerate(epochs_list):
            ep_steps = [d["step"] for d in train_data if d["epoch"] == ep]
            if ep_steps:
                ax1.axvspan(min(ep_steps), max(ep_steps),
                            alpha=0.3, color=colors[idx % len(colors)],
                            label=f"Epoch {ep}")

        ax1.plot(steps, losses, color="#2563eb", linewidth=0.8, alpha=0.6, label="Train loss (step)")

        # Smooth MA-50
        window = min(50, len(losses))
        if window > 1:
            ma = [sum(losses[max(0, j - window): j + 1]) / min(j + 1, window) for j in range(len(losses))]
            ax1.plot(steps, ma, color="#1e40af", linewidth=2, label=f"MA-{window}")

        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Train Loss per Step")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Chưa có dữ liệu train loss", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Train Loss per Step")

    # ── Plot 2: Validation loss theo step ────────────────────────────────────
    ax2 = axes[1]
    if val_data:
        val_steps  = [d["step"] for d in val_data]
        val_losses = [d["loss"] for d in val_data]
        ax2.plot(val_steps, val_losses, "o-", color="#dc2626", linewidth=2,
                 markersize=6, markerfacecolor="white", markeredgewidth=2, label="Val loss")
        for x, y in zip(val_steps, val_losses):
            ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), fontsize=7, ha="center", color="#dc2626")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation Loss per Eval Step")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Chưa có dữ liệu val loss\n(chạy đến eval_steps=2000 mới có)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=9)
        ax2.set_title("Validation Loss per Eval Step")

    # ── Plot 3: Epoch summary ────────────────────────────────────────────────
    ax3 = axes[2]
    if epoch_data:
        ep_nums   = [d["epoch"] for d in epoch_data]
        ep_losses = [d["avg_train_loss"] for d in epoch_data]
        bars = ax3.bar(ep_nums, ep_losses, color=["#3b82f6", "#6366f1", "#8b5cf6"][:len(ep_nums)],
                       edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, ep_losses):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Avg Train Loss")
        ax3.set_title("Avg Train Loss per Epoch")
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax3.set_ylim(0, max(ep_losses) * 1.2)
        ax3.grid(True, axis="y", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Chưa hoàn thành epoch nào", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Avg Train Loss per Epoch")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Đã lưu biểu đồ: {save_path}")

    plt.show()
    return fig


def print_summary(metrics: dict):
    """In tóm tắt ra terminal."""
    train_data = metrics.get("train_loss", [])
    val_data   = metrics.get("val_loss", [])
    epoch_data = metrics.get("epoch_summary", [])

    print("\n" + "=" * 55)
    print("  TRAINING SUMMARY")
    print("=" * 55)

    if epoch_data:
        print("\n📊 Epoch Summary:")
        for ep in epoch_data:
            print(f"   Epoch {ep['epoch']:>2}: avg_train_loss = {ep['avg_train_loss']:.6f}")

    if val_data:
        print("\n📉 Validation Loss History:")
        for v in val_data:
            print(f"   Epoch {v['epoch']} | Step {v['step']:>6}: val_loss = {v['loss']:.6f}")
        best = min(val_data, key=lambda x: x["loss"])
        print(f"\n   🏆 Best val_loss = {best['loss']:.6f} @ epoch {best['epoch']} step {best['step']}")

    if train_data:
        total_steps = max(d["step"] for d in train_data)
        last_loss   = train_data[-1]["loss"]
        print(f"\n📈 Train steps logged: {len(train_data)} | Last step: {total_steps} | Last loss: {last_loss:.6f}")

    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics from metrics.json")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path to metrics.json. Nếu không truyền, tự tìm file mới nhất.")
    parser.add_argument("--base_dir", type=str, default="./outputs/internvl3_2b",
                        help="Base output dir để auto-search metrics.json")
    parser.add_argument("--save", type=str, default=None,
                        help="Lưu biểu đồ ra file PNG (vd: plot.png)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Chỉ in summary, không vẽ biểu đồ")
    args = parser.parse_args()

    metrics_path = args.metrics or find_latest_metrics(args.base_dir)
    print(f"[INFO] Đọc metrics từ: {metrics_path}")
    metrics = load_metrics(metrics_path)

    print_summary(metrics)

    if not args.no_plot:
        run_dir = os.path.basename(os.path.dirname(metrics_path))
        plot_metrics(metrics, save_path=args.save, title_prefix=f"[{run_dir}] ")


if __name__ == "__main__":
    main()
