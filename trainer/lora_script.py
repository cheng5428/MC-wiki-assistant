import os
import sys
import subprocess

def run_lora_training():
    """
    运行LoRA训练脚本
    """
    # 构建命令行参数
    cmd = [
        "python", "train_lora.py",
        "--lora_name", "lora_mc",
        "--data_path", "../dataset/mc_data.jsonl",
        "--out_dir", "../out",
        "--epochs", "40",
        "--save_epoch_interval", "10",
        # "--use_wandb", 
        # "--wandb_project", "MC-wiki-assistant",   # 如果需要使用wandb进行实验跟踪
    ]
    
    print("启动LoRA训练...")
    print(f"执行命令: {' '.join(cmd)}")
    
    # 设置环境变量(如需要)
    env = os.environ.copy()
    
    # 执行命令
    try:
        process = subprocess.run(
            cmd,
            env=env,
            check=True,
            text=True
        )
        print("训练完成!")
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"训练过程中出现错误: {e}")
        return e.returncode

if __name__ == "__main__":
    # 获取脚本当前所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 切换到脚本所在目录，确保相对路径正确
    os.chdir(script_dir)
    
    # 运行训练
    sys.exit(run_lora_training())