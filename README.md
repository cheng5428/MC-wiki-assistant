# MC-wiki-assistant

## Getting Started

### Setup
1. Clone this repository
2. Download the MiniMind2-Small model's `model.safetensors` file and place it in the `MiniMind2-Small` directory
    - Alternatively: `git clone https://hf-mirror.com/jingyaogong/MiniMind2-Small` to replace the MiniMind2-Small directory

### Training LoRA Models
1. Run `scripts/convert_model.py` to convert the `.safetensors` format to `.pth` format
2. Execute `trainer/lora_script.py` to begin training (see file for details)

### Testing
- To test a LoRA model: `python eval_model.py --lora_name [lora_name]`
  - Example: `python eval_model.py --lora_name 'lora_mc_40'` (no need to include the '512' suffix)
- To test the web UI: 
  -'cd scripts'
  -`streamlit run web_demo.py`