import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import RadarDataset
from model import RadarModel
from utils import get_test_samples, get_tta_views


def main():
    print("===============================")
    print("      RADAR INFERENCE          ")
    print("===============================")

    val_samples, val_ids = get_test_samples(CFG.VAL_DIR)
    test_samples, test_ids = get_test_samples(CFG.TEST_DIR)

    if len(val_samples) == 0 and len(test_samples) == 0:
        print("No data available for inference (val/test). Exiting program.")
        return

    all_samples = val_samples + test_samples
    all_ids = val_ids + test_ids

    test_ds = RadarDataset(all_samples, labels=None, train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.num_workers
    )

    model_path = os.path.join(CFG.CHECKPOINT_DIR, f"{CFG.best_model}.pth")
    if not os.path.exists(model_path):
        print(f"Error: Weights file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = RadarModel(CFG.NUM_CLASSES).to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    all_predictions = []
    loader = tqdm(test_loader, desc="Inference") if CFG.USE_TQDM else test_loader

    with torch.no_grad():
        for x in loader:
            x = x.to(CFG.device)
            views = get_tta_views(x)
            prob_sum = None
            for view in views:
                logits = model(view)
                probs = torch.softmax(logits, dim=1)
                if prob_sum is None:
                    prob_sum = probs
                else:
                    prob_sum += probs
            avg_prob = prob_sum / len(views)
            preds = avg_prob.argmax(dim=1).cpu().numpy()
            all_predictions.extend(preds)

    submission_df = pd.DataFrame({
        "id": all_ids,
        "pred": all_predictions
    })
    submission_df["id"] = submission_df["id"].str.replace(r"^SAMPLE_", "", regex=True)

    out_file = os.path.join(CFG.RESULTS_DIR, "submission.csv")
    submission_df.to_csv(out_file, index=False)
    print(f"\nFinish Inferencing! Results have been saved at: {out_file}")


if __name__ == "__main__":
    main()