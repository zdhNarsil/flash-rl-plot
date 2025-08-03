import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os



# metrics = ["training/rollout_probs_diff_max", "training/rollout_probs_diff_min", 
#            "train/vllm_kl", "val-core/math_dapo/acc/mean@32"]

FIG_SIZE = (8, 6)


def plot_metrics(combined_df, metrics_to_plot=None, save_dir="./", label=None, max_step=None):
    """
    Args:
        combined_df: 拼接后的DataFrame
        metrics_to_plot: 要绘制的指标列表，如果为None则绘制所有数值型指标
        save_dir: 图片保存目录
        label: 图例标签
        max_step: _step的最大值，超过这个值的数据点不会被绘制
    """
    if '_step' not in combined_df.columns:
        print("'_step' column not found in combined data.")
        return
    
    # 获取数值型列（排除_step, _timestamp等系统列）
    numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
    if metrics_to_plot is None:
        metric_columns = [col for col in numeric_columns if not col.startswith('_')]
    else:
        metric_columns = [col for col in metrics_to_plot if col in combined_df.columns]
    print(f"Found numeric metrics: {metric_columns}")
    if len(metric_columns) == 0:
        print("No metrics found to plot.")
        return
    
    # colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
    # run_names = combined_df['run_name'].unique()
    
    # 为每个指标创建单独的图
    for metric in metric_columns:
        plt.figure(figsize=FIG_SIZE)
        
        # 整体曲线（所有数据点连接）
        non_na_data = combined_df.dropna(subset=[metric])
        if max_step is not None:
            non_na_data = non_na_data[non_na_data['_step'] <= max_step]
        if not non_na_data.empty:
            plt.plot(non_na_data['_step'], non_na_data[metric], 
                    color='black', linewidth=1, linestyle='-',
                    label=label)
        
        plt.xlabel('Step', fontsize=10)
        # plt.ylabel(metric, fontsize=10)
        plt.title(f'{metric}', fontsize=10)
        
        # Set y-axis limit for training/rollout_probs_diff_max
        if metric == 'training/rollout_probs_diff_max':
            # plt.ylim(0.99997, 1.00001)
            plt.ylim(0., 1.1)
        # Set y-axis limit for training/rollout_probs_diff_min  
        elif metric == 'training/rollout_probs_diff_min':
            plt.ylim(-1., 0.)
            
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        safe_filename = metric.replace('/', '_').replace('@', '_at_')
        save_path = f"{save_dir}/{safe_filename}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot for {metric} to {save_path}")
        
        # plt.show()
        plt.close()  # 关闭当前图形以释放内存

def plot_comparison_metrics(combined_df, run3_df, metrics_to_plot=None, 
        save_dir="./", label1=None, label2=None, max_step=None, file_prefix="comparison"):
    """
    绘制对比图：combined run1+run2 vs run3
    
    Args:
        combined_df: 拼接后的run1+run2数据
        run3_df: run3的数据
        metrics_to_plot: 要绘制的指标列表
        save_dir: 图片保存目录
        label1: 第一条曲线的标签
        label2: 第二条曲线的标签
        max_step: _step的最大值，超过这个值的数据点不会被绘制
    """
    if '_step' not in combined_df.columns or '_step' not in run3_df.columns:
        print("'_step' column not found in data.")
        return
    
    # 获取要绘制的指标
    if metrics_to_plot is None:
        numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
        metric_columns = [col for col in numeric_columns if not col.startswith('_')]
    else:
        metric_columns = [col for col in metrics_to_plot if col in combined_df.columns and col in run3_df.columns]
    print(f"Found comparison metrics: {metric_columns}")
    
    if len(metric_columns) == 0:
        print("No metrics found to plot.")
        return
    
    # 为每个指标创建对比图
    for metric in metric_columns:
        plt.figure(figsize=FIG_SIZE)
        
        # 绘制combined run1+run2曲线
        combined_non_na = combined_df.dropna(subset=[metric])
        if max_step is not None:
            combined_non_na = combined_non_na[combined_non_na['_step'] <= max_step]
        if not combined_non_na.empty:
            plt.plot(combined_non_na['_step'], combined_non_na[metric], 
                    color='#1f77b4', linewidth=2, alpha=0.8,
                    label=label1)
        
        # 绘制run3曲线
        run3_non_na = run3_df.dropna(subset=[metric])
        if max_step is not None:
            run3_non_na = run3_non_na[run3_non_na['_step'] <= max_step]
        if not run3_non_na.empty:
            plt.plot(run3_non_na['_step'], run3_non_na[metric], 
                    color='#ff7f0e', linewidth=2, alpha=0.8,
                    label=label2)
        
        plt.xlabel('Step', fontsize=10)
        # plt.ylabel(metric, fontsize=10)
        plt.title(f'{metric}', fontsize=10)
        
        # Set y-axis limit for training/rollout_probs_diff_max
        if metric == 'training/rollout_probs_diff_max':
            plt.ylim(0., 1.1)
        # Set y-axis limit for training/rollout_probs_diff_min  
        elif metric == 'training/rollout_probs_diff_min':
            plt.ylim(-1., 0.)
            
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存对比图
        safe_filename = metric.replace('/', '_').replace('@', '_at_')
        save_path = f"{save_dir}/{file_prefix}_{safe_filename}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot for {metric} to {save_path}")
        
        # plt.show()
        plt.close()



def plot_four_curves_comparison(dataframes, metrics_to_plot=None, labels=None, 
                               save_dir="./", max_step=None, file_prefix="comparison"):
    """
    绘制四条曲线的对比图
    
    Args:
        dataframes: 包含四个DataFrame的列表
        metrics_to_plot: 要绘制的指标列表
        labels: 四条曲线的标签列表
        save_dir: 图片保存目录
        max_step: _step的最大值，超过这个值的数据点不会被绘制
        file_prefix: 文件名前缀
    """
    if len(dataframes) != 4 or len(labels) != 4:
        print("Error: Need exactly 4 dataframes and 4 labels")
        return
    
    # 检查所有DataFrame是否都有_step列
    for i, df in enumerate(dataframes):
        if '_step' not in df.columns:
            print(f"'_step' column not found in dataframe {i}")
            return
    
    # 获取要绘制的指标
    if metrics_to_plot is None:
        numeric_columns = dataframes[0].select_dtypes(include=['float64', 'int64']).columns
        metric_columns = [col for col in numeric_columns if not col.startswith('_')]
    else:
        metric_columns = [col for col in metrics_to_plot if all(col in df.columns for df in dataframes)]
    
    print(f"Found comparison metrics: {metric_columns}")
    
    if len(metric_columns) == 0:
        print("No metrics found to plot.")
        return
    
    # 定义四种颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝色、橙色、绿色、红色
    
    # 为每个指标创建对比图
    for metric in metric_columns:
        plt.figure(figsize=FIG_SIZE)
        
        # 绘制四条曲线
        for i, (df, label, color) in enumerate(zip(dataframes, labels, colors)):
            non_na_data = df.dropna(subset=[metric])
            if max_step is not None:
                non_na_data = non_na_data[non_na_data['_step'] <= max_step]
            if not non_na_data.empty:
                plt.plot(non_na_data['_step'], non_na_data[metric], 
                        color=color, linewidth=2, alpha=0.8,
                        label=label)
        
        plt.xlabel('Step', fontsize=10)
        plt.title(f'{metric}', fontsize=10)
        
        # Set y-axis limits based on metric
        if metric == 'training/rollout_probs_diff_max':
            plt.ylim(0., 1.1)
        elif metric == 'training/rollout_probs_diff_min':
            plt.ylim(-1., 0.)
        
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存对比图
        safe_filename = metric.replace('/', '_').replace('@', '_at_')
        save_path = f"{save_dir}/{file_prefix}_{safe_filename}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved four-curve comparison plot for {metric} to {save_path}")
        
        plt.close()
