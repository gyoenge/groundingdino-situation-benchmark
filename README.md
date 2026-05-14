<!-- # Net Challenge Project - **Dataset Process** 

## GroundingDINO Finetuning Dataset Processing/Preparing   -->

# GroundingDINO Situation Benchmark 

This repository contains **benchmark construction code for evaluating prompt-based situation understanding capabilities**, data preprocessing scripts, and knife safety detection dataset generation codes.

<!-- This repository contains the **code used to build the custom dataset for fine-tuning the GroundingDINO model**.  -->

<!-- About
A project from the K-Digital Net Challenge (2023). This repository contains benchmark construction code for evaluating prompt-based situation understanding capabilities, data preprocessing scripts, and knife safety detection dataset generation codes. -->

<!-- The main code for this project is located in the [GroundingDINO Finetune](https://github.com/gyoenge/net-challenge-groundingdino-finetune) repository, based on [Original GroundingDINO Finetune Pipeline Opensource](https://github.com/Asad-Ismail/Grounding-Dino-FineTuning). Please visit it for more details about the project.  -->

### Environment: 

Install dependencies. 
```bash 
conda create -n groundingdino python=3.10 # check version 
conda activate groundingdino
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # check version
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e . --no-build-isolation 
```

Download pre-trained model weights.
```bash 
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../.. 
```

### Run: 

Inside the project root, 
First run for dataset generation: 
```bash 
python gen_dataset.py \
  --output_dir data \
  --num_samples 10 \
  --device cuda
```

Then run for evaluation: 
```bash 
python main.py \
  --annotations data/annotations.json \
  --predictions outputs/predictions.json \
  --report outputs/evaluation_report.json 
```

---

### Example annotation: 

    [
    {
        "image_id": "sample_001",
        "image_path": "data/images/sample_001.jpg",
        "category": "dangerous_knife",
        "prompts": [
        {
            "prompt_id": "p001",
            "prompt": "person",
            "type": "object",
            "expected": "positive"
        },
        {
            "prompt_id": "p002",
            "prompt": "knife",
            "type": "object",
            "expected": "positive"
        },
        {
            "prompt_id": "p003",
            "prompt": "person with knife",
            "type": "composition",
            "expected": "positive"
        },
        {
            "prompt_id": "p004",
            "prompt": "a man threatening someone with a knife",
            "type": "situation",
            "expected": "positive"
        },
        {
            "prompt_id": "p005",
            "prompt": "a chef cooking with a knife",
            "type": "situation",
            "expected": "negative"
        }
        ],
        "annotations": [
        {
            "label": "person",
            "bbox": [100, 80, 300, 420]
        },
        {
            "label": "knife",
            "bbox": [260, 200, 330, 250]
        }
        ]
    }
    ]
