# MC-wiki-assistant

clone该仓库后，先将MiniMind2-Small模型的model.safetensors放到MiniMind2-Small目录下，也可以选择git clone https://hf-mirror.com/jingyaogong/MiniMind2-Small替换MiniMind2-Small目录
如果想要训练lora，请先运行scripts/convert_model.py，将.safetensors格式转换为.pth格式，接着运行trainer/lora_script.py即可（参见该文件）即可
想要测试lora模型请在命令行运行 python eval_model.py --lora_name ['lora名字'](例如：'lora_mc_40'，注意后缀512不需要)
想要测试网站的ui界面，请运行 streamlit run scripts/web_demo.py