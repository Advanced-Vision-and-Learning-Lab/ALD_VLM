# extract_features.py
"""
Extracts WSI patch features using Vision-Language models (PLIP/CONCH).
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================
# AUTO-OFFLINE SETUP & CACHING
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "models" / "cache_for_HF"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_has_files(cache_root: Path) -> bool:
    if not cache_root.exists():
        return False
    for p in cache_root.rglob("*"):
        if p.is_file() and p.stat().st_size > 0:
            return True
    return False

os.environ.setdefault("HF_HOME", str(CACHE_DIR))

if _cache_has_files(CACHE_DIR):
    logging.info("Local HF cache found. Automatically switching to OFFLINE mode.")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
else:
    logging.info("Local HF cache not found. Running in ONLINE mode to download models.")


# ==========================================
# TEXTUAL ENSEMBLE PROMPTS
# ==========================================
TEMPLATES = [
    "An H&E stained liver patch showing {}.",
    "Microscopic view of {}.",
    "Liver histology demonstrating {}.",
    "Pathology slide revealing {}."
]

CONCEPTS = {
    "Non-Drinking": [
        "normal liver parenchyma", "healthy hepatocytes", "intact lobular architecture",
        "an absence of steatosis", "an absence of inflammation"
    ],
    "Drinking": [
        "macrovesicular steatosis", "microvesicular lipid accumulation",
        "hepatocyte ballooning degeneration", "neutrophilic lobular inflammation",
        "perisinusoidal fibrosis"
    ]
}

CLASS_PROMPTS = {"Non-Drinking": [], "Drinking": []}
for label, concepts in CONCEPTS.items():
    for concept in concepts:
        for template in TEMPLATES:
            CLASS_PROMPTS[label].append(template.format(concept))

CLASSES_ORDER = ["Non-Drinking", "Drinking"]


# ==========================================
# DATASETS & HELPERS
# ==========================================
def extract_tensor(outputs: any) -> torch.Tensor:
    """Extracts raw tensor from various model output formats."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs

class PatchFeatureDataset(Dataset):
    """Dataset wrapper for extracted WSI patches."""
    def __init__(self, patches_tensor: torch.Tensor):
        self.patches = patches_tensor

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx].permute(1, 2, 0).numpy()


# ==========================================
# EXTRACTION PIPELINE
# ==========================================
def extract_text_embeddings(model, processor, tokenizer, device: str, output_dir: Path, model_choice: str):
    """Encodes standard prompts using the selected VL model."""
    logging.info("Extracting Text Embeddings...")
    model.eval()
    
    text_features_list = []
    specific_features_dict = {} 
    
    with torch.no_grad():
        for cls in CLASSES_ORDER:
            prompts = CLASS_PROMPTS[cls]
            
            if model_choice == "PLIP":
                inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = model.get_text_features(**inputs)
                
            elif model_choice == "CONCH":
                tokenized_dict = tokenizer(
                    prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
                )
                inputs = tokenized_dict["input_ids"].to(device)
                outputs = model.encode_text(inputs)
            
            features = extract_tensor(outputs)
            features = F.normalize(features, p=2, dim=-1) 
            
            specific_features_dict[cls] = {
                "prompts": prompts,
                "features": features.cpu() 
            }
            
            avg_feature = features.mean(dim=0)
            avg_feature = F.normalize(avg_feature, p=2, dim=-1)
            text_features_list.append(avg_feature.cpu().numpy())
            
    # Save Ensemble
    T = np.stack(text_features_list)
    out_path = output_dir / "text_prototypes.npy"
    np.save(out_path, T)
    logging.info(f"Ensemble text prototypes saved to {out_path} with shape {T.shape}")
    
    # Save Specifics
    spec_out_path = output_dir / "specific_text_features.pt"
    torch.save(specific_features_dict, spec_out_path)
    logging.info(f"Specific text prototypes saved to {spec_out_path}")


def extract_image_embeddings(model, processor, device: str, prepared_dir: Path, output_dir: Path, batch_size: int, model_choice: str):
    """Encodes local WSI patches into feature vectors."""
    logging.info("Extracting Image Embeddings...")
    model.eval()
    
    for split in ["train", "val", "test"]:
        split_dir = prepared_dir / split
        out_split_dir = output_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        
        pt_files = list(split_dir.glob("*.pt"))
        if not pt_files:
            continue
            
        logging.info(f"Processing {split} split ({len(pt_files)} slides)...")
        for pt_path in pt_files:
            data = torch.load(pt_path, map_location="cpu")
            patches = data["patches"]
            slide_name = data["slide_name"]
            raw_category = data["category"]
            binary_category = "Non-Drinking" if raw_category == "ND" else "Drinking"
            
            dataset = PatchFeatureDataset(patches)
            
            def collate_fn(batch):
                if model_choice == "PLIP":
                    return processor(images=batch, return_tensors="pt")
                elif model_choice == "CONCH":
                    batch_np = np.stack(batch)
                    batch_ts = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float()
                    
                    if batch_ts.max() > 1.0:
                        batch_ts /= 255.0
                        
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
                    batch_ts = (batch_ts - mean) / std
                    
                    return {"pixel_values": batch_ts}
                    
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
            slide_features = []
            
            pbar = tqdm(dataloader, desc=f"  Extracting {slide_name}", leave=False)
            with torch.no_grad():
                for batch in pbar:
                    pixel_values = batch["pixel_values"].to(device)
                    
                    if model_choice == "PLIP":
                        outputs = model.get_image_features(pixel_values=pixel_values)
                    elif model_choice == "CONCH":
                        outputs = model.encode_image(pixel_values)
                        
                    features = extract_tensor(outputs)
                    features = F.normalize(features, p=2, dim=-1)
                    slide_features.append(features.cpu().numpy())
            
            F_matrix = np.concatenate(slide_features, axis=0)
            
            out_file = out_split_dir / f"{pt_path.stem}_features.pt"
            feature_payload = {
                "features": torch.from_numpy(F_matrix),
                "coords": data["coords"],
                "subject_id": data["subject_id"],
                "slide_name": slide_name,
                "raw_category": raw_category,
                "binary_category": binary_category,
                "split": split
            }
            torch.save(feature_payload, out_file)
            logging.info(f"Saved: {slide_name} -> {out_file.name} | Shape: {F_matrix.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract WSI features")
    parser.add_argument('--model', type=str, choices=["PLIP", "CONCH"], default="CONCH", help="Choose the vision-language model")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for extraction")
    args, _ = parser.parse_known_args() 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    prepared_dir = PROJECT_ROOT / "dataset" / "prepared_dataset"
    output_dir = PROJECT_ROOT / "outputs" / f"features_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = None
    if args.model == "PLIP":
        logging.info("Loading PLIP model...")
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("vinid/plip", cache_dir=str(CACHE_DIR)).to(device)
        processor = CLIPProcessor.from_pretrained("vinid/plip", cache_dir=str(CACHE_DIR))
        
    elif args.model == "CONCH":
        logging.info("Loading CONCH model from local checkpoint...")
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        conch_checkpoint = PROJECT_ROOT / "models" / "CONCH" / "pytorch_model.bin"
        
        if not conch_checkpoint.exists():
            raise FileNotFoundError(f"Could not find CONCH weights at {conch_checkpoint}")
            
        model, processor = create_model_from_pretrained('conch_ViT-B-16', str(conch_checkpoint))
        model = model.to(device)
        tokenizer = get_tokenizer()
    
    extract_text_embeddings(model, processor, tokenizer, device, output_dir, args.model)
    extract_image_embeddings(model, processor, device, prepared_dir, output_dir, args.batch_size, args.model)
    logging.info(f"{args.model} Extraction Complete!")

if __name__ == "__main__":
    main()

