import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_ratio_from_folder(folder_name):
    """Extract labeled ratio from folder name."""
    if folder_name == "PREDICTION":
        return 0.5  # default
    match = re.search(r'(\d+\.\d+|\d+)', folder_name)
    if match:
        return float(match.group(1))
    return None

def get_mean_miou(csv_path):
    """Reads eval_metrics.csv and returns MeanIoU value."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if 'MeanIoU' in col or 'meaniou' in col.lower():
            return df[col].values[0]
    # fallback: try IoU column if MeanIoU not found
    for col in df.columns:
        if 'iou' in col.lower():
            return df[col].values[0]
    return None

def get_final_training_loss(loss_csv_path):
    """Reads losses.csv and returns the last epoch Total loss."""
    if not os.path.exists(loss_csv_path):
        return None
    df = pd.read_csv(loss_csv_path)
    if 'Total' in df.columns:
        return df['Total'].values[-1]  # last epoch
    return None

def main(predictions_root, models_root, output_dir="."):
    folders = sorted([f for f in os.listdir(predictions_root) if f.startswith("PREDICTION")])
    ratios, miou_vals, loss_vals = [], [], []

    for folder in folders:
        #pred_path = os.path.join(predictions_root, folder, "flame")
        pred_path = os.path.join(predictions_root, folder, "meta")
        if not os.path.exists(pred_path):
            continue
        csv_pred = os.path.join(pred_path, "eval_metrics.csv")
        ratio = extract_ratio_from_folder(folder)
        miou = get_mean_miou(csv_pred)

        # Find corresponding model folder
        model_folder = folder.replace("PREDICTION", "MODEL_FILE")
        model_path = os.path.join(models_root, model_folder, "losses.csv")
        final_loss = get_final_training_loss(model_path)

        if miou is not None and final_loss is not None:
            ratios.append(ratio)
            miou_vals.append(miou)
            loss_vals.append(final_loss)

    if not ratios:
        print("? No valid metrics found.")
        return

    # Sort by ratio
    sorted_pairs = sorted(zip(ratios, miou_vals, loss_vals))
    ratios, miou_vals, loss_vals = zip(*sorted_pairs)

    os.makedirs(output_dir, exist_ok=True)

    # Plot mIoU vs Labeled Ratio
    plt.figure(figsize=(8,6))
    plt.plot(ratios, miou_vals, marker='o', linewidth=2, color='royalblue')
    plt.xlabel("Labeled Dataset Ratio")
    plt.ylabel("Mean IoU (All Classes)")
    plt.title("mIoU vs Labeled Dataset Ratio")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(ratios)
    plt.tight_layout()
    miou_plot_path = os.path.join(output_dir, "mIoU_vs_labeled_ratio.png")
    plt.savefig(miou_plot_path)
    print(f"? Saved mIoU plot to {miou_plot_path}")
    plt.close()

    # Plot Training Loss vs Labeled Ratio
    plt.figure(figsize=(8,6))
    plt.plot(ratios, loss_vals, marker='o', linewidth=2, color='crimson')
    plt.xlabel("Labeled Dataset Ratio")
    plt.ylabel("Training Total Loss (Last Epoch)")
    plt.title("Training Loss vs Labeled Dataset Ratio")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(ratios)
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, "loss_vs_labeled_ratio.png")
    plt.savefig(loss_plot_path)
    print(f"? Saved training loss plot to {loss_plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=str, required=True,
                        help="Path to folder containing PREDICTION* folders")
    parser.add_argument("--models-root", type=str, required=True,
                        help="Path to folder containing corresponding MODEL_FILE* folders")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Folder to save the plots")
    args = parser.parse_args()
    main(args.predictions_root, args.models_root, args.output_dir)
