import os
import json
import torch
from PIL import Image
from torchvision.transforms import v2
from peft import LoraConfig, get_peft_model
import evaluate

from models import CaptioningModel
from tokenizer import build_tokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "../datasets/vizwiz"
    
    # Target images to evaluate
    target_data = [
        {
            "image_id": 456,
            "file_name": "VizWiz_train_00004732.jpg",
            "references": [
                "wrinkled Royal Blue Sheet Fabric Close up on Steam",
                "Blue looks good on cotton fabric like this.",
                "some sort of blue fabric that is gently cleaned",
                "A blue item of clothing has a seam.",
                "A large puffy blue material with a line running through"
            ]
        },
        {
            "image_id": 102,
            "file_name": "VizWiz_train_00001046.jpg",
            "references": [
                "A yellow plastic bag of a nut mix that says Almond Energy.",
                "A hand holding a yellow bag of Almonds",
                "Quality issues are too severe to recognize visual content.",
                "Quality issues are too severe to recognize visual content.",
                "A person holding a bag of some sort of almond snack."
            ]
        },
        {
            "image_id": 1126,
            "file_name": "VizWiz_train_00011613.jpg",
            "references": [
                "A hand touches a black cat who sits on a closed toilet.",
                "A hand petting an animal that is sitting on top of a toilet.",
                "A person is rubbing a medium size black cat.",
                "A black cat sitting on top of a white toilet.",
                "Someone holding something furry, probably a black cat, the furry object is sitting on a closed toilet seat."
            ]
        },
        {
            "image_id": 1003,
            "file_name": "VizWiz_train_00010484.jpg",
            "references": [
                "A washer panel displays that items are clean.",
                "Quality issues are too severe to recognize visual content.",
                "A white appliance is turned on to clean.",
                "a panel for a kitchen appliance that contains buttons to operate",
                "The green light for clean is lit on a control panel."
            ]
        },
        {
            "image_id": 914,
            "file_name": "VizWiz_train_00009623.jpg",
            "references": [
                "a hand on top of a can of salmon that is on a marble kitchen counter",
                "A tin can of Sno-Tip brand Wild Alaska chum salmon with someone's hand on top of it.",
                "A can of Snow tip wild Alaska chum salmon.",
                "A person has a can of fish on the counter.",
                "Aluminum can with salmon placed on granite countertop."
            ]
        },
        {
            "image_id": 571,
            "file_name": "VizWiz_train_00005985.jpg",
            "references": [
                "a white paper labelled with cuisine chicken marsala",
                "A box of Lean Cuisine Chicken Marsala on a counter.",
                "a paper box pack of lean cuisine culinary collection",
                "Package of frozen food lying flat on a horizontal surface.",
                "A Lean Cuisine Chicken Marsala is laying on a counter top."
            ]
        },
        {
            "image_id": 419,
            "file_name": "VizWiz_train_00004246.jpg",
            "references": [
                "a person holding a plastic package of meat in their hands",
                "a person holding a sausage log that is showing the nutrition facts",
                "A person's hand holding a tube of sausage.",
                "A hand holding a tube of a meat product",
                "Someone holding a package of some meat, nutritional information or list of ingredients showing."
            ]
        },
        {
            "image_id": 2233,
            "file_name": "VizWiz_train_00022288.jpg",
            "references": [
                "A clear decorative vase sitting on a stand with a white lacy curtain behind.",
                "A glass bowl with a plant in it on a table behind a couch.",
                "A crystal vase with stems in it in front of a lace curtain next to a sofa.",
                "The vase is sitting directly behind the sofa and in front of the lacy curtain.",
                "a glass container holding some sort of flower that's green"
            ]
        },
        {
            "image_id": 356,
            "file_name": "VizWiz_train_00003544.jpg",
            "references": [
                "Appears to  be a picture of a cell phone",
                "A small Nokia brand cellular phone, the screen is off.",
                "A silver colored cell phone is laying on a white, gold streaked counter.",
                "A small old school cell phone on a counter top",
                "Small Nokia phone facing right side up on a counter top"
            ]
        },
        {
            "image_id": 1728,
            "file_name": "VizWiz_train_00017388.jpg",
            "references": [
                "A pair of marbles with strings coming out of them laying on a blanket.",
                "Two gold medallions handing from string laying on a blue fabric surface.",
                "Two small, yellow, marble-like balls with white cords coming out of them.",
                "two strings with a gold bead on the end of each.",
                "Two yellow balls connected to a white string."
            ]
        }
    ]

    # 1. Rebuild Tokenizer for vocab mapping
    print("Loading tokenizer...")
    train_ann = os.path.join(data_root, 'annotations', 'train.json')
    cache_dir = os.path.join(data_root, 'tokenizer_cache')
    tokenizer = build_tokenizer('subword', train_ann, cache_dir)
    
    # Load METEOR metric
    print("Loading METEOR metric...")
    meteor_metric = evaluate.load('meteor')
    
    # 2. Setup standard image transformations
    img_proc = torch.nn.Sequential(
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ).to(device)

    # 3. Load 0.8B Model
    print("Initializing Qwen 0.8B Model architecture...")
    model = CaptioningModel(
        encoder_name='clip', 
        decoder_type='qwen',
        decoder_model_name='Qwen/Qwen3.5-0.8B-Base',
        vocab_size=tokenizer.vocab_size,
        hidden_dim=768 # Projection dimension for CLIP base
    ).to(device)
    
    # Apply LoRA configuration to the decoder
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj" 
    ]
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model.decoder.qwen = get_peft_model(model.decoder.qwen, config)
    
    # Load weights
    weights_path = "./outputs/clip_qwen_0.8B/best_metric_model.pt"
    print(f"Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.eval()

    print("\n" + "="*80)
    print("0.8B INFERENCE ON TARGET IMAGES WITH METEOR")
    print("="*80)

    train_img_dir = os.path.join(data_root, 'train')
    results = []
    
    # Lists to store all predictions and references for the aggregate score
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for item in target_data:
            img_path = os.path.join(train_img_dir, item['file_name'])
            
            if not os.path.exists(img_path):
                print(f"[!] Warning: Image not found at {img_path}. Skipping.")
                continue

            # Process Image
            image = Image.open(img_path).convert("RGB")
            img_tensor = img_proc(image).unsqueeze(0).to(device)
            
            # Generate Caption
            gen = model.generate(img_tensor, tokenizer.max_len - 1, tokenizer.sos_idx, tokenizer.eos_idx)
            pred_08b = tokenizer.decode(gen[0].cpu().tolist()).strip()
            
            # Compute instance-level METEOR
            score = meteor_metric.compute(predictions=[pred_08b], references=[item['references']])['meteor']
            
            # Store data
            item["0.8B_prediction"] = pred_08b
            item["0.8B_meteor_score"] = score
            results.append(item)
            
            all_predictions.append(pred_08b)
            all_references.append(item['references'])
            
            print(f"\nImage: {item['file_name']}")
            print("-" * 40)
            print(f"0.8B Pred:   {pred_08b}")
            print(f"Reference 1: {item['references'][0]}")
            print(f"METEOR:      {score * 100:.2f}%")
            print("="*80)

    # Compute overall METEOR for this subset
    if all_predictions:
        overall_meteor = meteor_metric.compute(predictions=all_predictions, references=all_references)['meteor']
        print(f"\n>>> OVERALL METEOR SCORE FOR SUBSET: {overall_meteor * 100:.2f}% <<<")
        
        # Add summary to the output JSON
        summary = {
            "summary": {
                "total_images": len(all_predictions),
                "overall_meteor_score": overall_meteor
            },
            "results": results
        }
    else:
        summary = {"results": results}

    # Save output to JSON
    output_file = "./outputs/clip_qwen_0.8B/subset_predictions.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Completed! Full predictions saved to {output_file}")

if __name__ == "__main__":
    main()