import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils_draw import plot_metrics, plot_comparison_metrics, plot_four_curves_comparison


os.environ["WANDB_BASE_URL"] = "https://microsoft-research.wandb.io"
api = wandb.Api()

metrics_dapo = ["training/rollout_probs_diff_max", "training/rollout_probs_diff_min", 
           "train/vllm_kl", "val-core/math_dapo/acc/mean@32"]

metrics_gsm8k = ["training/rollout_probs_diff_max", "training/rollout_probs_diff_min", 
           "training/vllm_kl", "val-core/openai/gsm8k/reward/mean@1"]

FIG_SIZE = (8, 6)

def get_run_data(run, metrics=None):
    """获取单个run的数据"""
    print(f"Run name: {run.name}")
    # print(f"Run tags: {run.tags}")
    # print(f"Run config: {run.config}")
    if metrics:
        history_df = run.history(keys=metrics, pandas=True)
    else:
        history_df = run.history(pandas=True)
    
    print(f"Available metrics: {list(history_df.columns)}")
    return history_df

def concatenate_runs(df1, df2, run1_name, run2_name):
    """
    拼接两个run的数据
    """
    # 为每个DataFrame添加run_name列
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    df1_copy['run_name'] = run1_name
    df2_copy['run_name'] = run2_name    
    # 拼接两个DataFrame
    combined_df = pd.concat([df1_copy, df2_copy], ignore_index=True)
    return combined_df



def main_dapo():
    # flash-baseline-T1-cleanedX
    # run1 = api.run("lucliu/flash/7246460007.18494-7196d987-b98d")
    # run2 = api.run("lucliu/flash/7246254745.98941-7cffd556-a37f")
    # history_df1 = get_run_data(run1, metrics)
    # history_df2 = get_run_data(run2, metrics)
    # combined_df = concatenate_runs(history_df1, history_df2, run1.name, run2.name)
    # combined_df.to_csv("/Users/narsilzhang/Codes/FlashRL/data/flash-baseline-T1-cleanedX.csv", index=False)
    # print("Saved combined_df to /Users/narsilzhang/Codes/FlashRL/data/flash-baseline-T1-cleanedX.csv")
    print("\n=== Plotting flash-baseline-T1-cleanedX Metrics ===")
    history_df1 = pd.read_csv("/Users/narsilzhang/Codes/FlashRL/data/flash-baseline-T1-cleanedX.csv")
    plot_metrics(history_df1, metrics_dapo, save_dir="/Users/narsilzhang/Codes/FlashRL/figures")


    ## flash-bf16-fp32head-async-T1.0-cleanedX
    # run3 = api.run("lucliu/flash/p1gg794h")
    # history_df3 = get_run_data(run3, metrics)
    # history_df3.to_csv("/Users/narsilzhang/Codes/FlashRL/data/flash-bf16-fp32head-async-T1.0-cleanedX.csv", index=False)
    # print("Saved history_df3 to /Users/narsilzhang/Codes/FlashRL/data/flash-bf16-fp32head-async-T1.0-cleanedX.csv")

    history_df3 = pd.read_csv("/Users/narsilzhang/Codes/FlashRL/data/flash-bf16-fp32head-async-T1.0-cleanedX.csv")
    # plot_metrics(history_df3, metrics, save_dir="/Users/narsilzhang/Codes/FlashRL/figures")


    print("\n=== Plotting Comparison: flash-baseline-T1-cleanedX vs Run3 (lash-bf16-fp32head-async-T1.0-cleanedX) ===")
    plot_comparison_metrics(history_df1, history_df3, metrics_dapo, 
            save_dir="/Users/narsilzhang/Codes/FlashRL/figures",  # change this to your own dir
            label1="flash-baseline-T1-cleanedX", 
            label2="flash-bf16-fp32head-async-T1.0-cleanedX",
            max_step=50,
            file_prefix="comparison")


    ## flash-bf16-all-async-nclip2-sHack-T1-cleanedX
    # run4 = api.run("lucliu/flash/7246258127.79197-fe346ef2-af92")
    # history_df4 = get_run_data(run4, metrics)
    # history_df4.to_csv("/Users/narsilzhang/Codes/FlashRL/data/flash-bf16-all-async-nclip2-sHack-T1-cleanedX.csv", index=False)
    # print("Saved history_df4 to /Users/narsilzhang/Codes/FlashRL/data/flash-bf16-all-async-nclip2-sHack-T1-cleanedX.csv")
    history_df4 = pd.read_csv("/Users/narsilzhang/Codes/FlashRL/data/flash-bf16-all-async-nclip2-sHack-T1-cleanedX.csv")
    # plot_metrics(history_df4, metrics, save_dir="/Users/narsilzhang/Codes/FlashRL/figures", label="flash-bf16-all-async-nclip2-sHack-T1-cleanedX")

    print("\n=== Plotting Comparison: flash-baseline-T1-cleanedX vs Run4 (flash-bf16-all-async-nclip2-sHack-T1-cleanedX) ===")
    plot_comparison_metrics(history_df1, history_df4, metrics_dapo, 
            save_dir="/Users/narsilzhang/Codes/FlashRL/figures", 
            label1="flash-baseline-T1-cleanedX", 
            label2="flash-bf16-all-async-nclip2-sHack-T1-cleanedX",
            file_prefix="comparison")


def main_gsm8k():
    # gsm8k-PPO-Qwen2.5-0.5B-bf16-sync-case_study
    # lucliu/GSM8K-PPO-new/7jcy4uqx
    
    # gsm8k-PPO-Qwen2.5-0.5B-w8a8-redhat-sync-api5-case_study-378
    # lucliu/GSM8K-PPO-new/iyy3zio8
    
    # gsm8k-PPO-Qwen2.5-0.5B-w8a8-redhat-sHack-async-api5-nclip2-case_study
    # lucliu/GSM8K-PPO-new/ezheqy5a

    # gsm8k-PPO-Qwen2.5-0.5B-bf16-redhat-async-api5-nclip2-case_study-378
    # lucliu/GSM8K-PPO-new/ntagwwsg
    
    # Get data for all four runs
    run1 = api.run("lucliu/GSM8K-PPO-new/7jcy4uqx")
    run2 = api.run("lucliu/GSM8K-PPO-new/iyy3zio8")
    run3 = api.run("lucliu/GSM8K-PPO-new/ezheqy5a")
    run4 = api.run("lucliu/GSM8K-PPO-new/ntagwwsg")
    
    history_df1 = get_run_data(run1, metrics_gsm8k)
    history_df2 = get_run_data(run2, metrics_gsm8k)
    history_df3 = get_run_data(run3, metrics_gsm8k)
    history_df4 = get_run_data(run4, metrics_gsm8k)
    
    # Save dataframes
    history_df1.to_csv("/Users/narsilzhang/Codes/FlashRL/data/gsm8k-bf16-sync.csv", index=False)
    history_df2.to_csv("/Users/narsilzhang/Codes/FlashRL/data/gsm8k-w8a8-sync.csv", index=False)
    history_df3.to_csv("/Users/narsilzhang/Codes/FlashRL/data/gsm8k-w8a8-async.csv", index=False)
    history_df4.to_csv("/Users/narsilzhang/Codes/FlashRL/data/gsm8k-bf16-async.csv", index=False)
    
    print("Saved all GSM8K dataframes")
    
    # Plot individual metrics for each run
    # print("\n=== Plotting Individual GSM8K Metrics ===")
    # plot_metrics(history_df1, metrics_gsm8k, save_dir="/Users/narsilzhang/Codes/FlashRL/figures", label="gsm8k-bf16-sync")
    # plot_metrics(history_df2, metrics_gsm8k, save_dir="/Users/narsilzhang/Codes/FlashRL/figures", label="gsm8k-w8a8-sync")
    # plot_metrics(history_df3, metrics_gsm8k, save_dir="/Users/narsilzhang/Codes/FlashRL/figures", label="gsm8k-w8a8-async")
    # plot_metrics(history_df4, metrics_gsm8k, save_dir="/Users/narsilzhang/Codes/FlashRL/figures", label="gsm8k-bf16-async")
    
    # Plot comparison with all four curves in one plot
    print("\n=== Plotting Comparison: All Four GSM8K Runs ===")
    plot_four_curves_comparison([history_df1, history_df2, history_df3, history_df4], 
                               metrics_gsm8k,
                               labels=["gsm8k-bf16-sync", "gsm8k-w8a8-sync", "gsm8k-w8a8-async", "gsm8k-bf16-async"],
                               save_dir="/Users/narsilzhang/Codes/FlashRL/figures",
                               file_prefix="gsm8k_comparison")


if __name__ == "__main__":
    # main_dapo()
    main_gsm8k()