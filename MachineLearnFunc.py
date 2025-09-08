## Custom Functions for Machine Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset,DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import ray
from ray import train,tune
from ray.tune import CLIReporter, Checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import inspect
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
# from umap import UMAP
# from scipy.stats import lognorm, poisson
# from collections import Counter
import time

import os
import tempfile
import json
############################
## hyperparameter tuning ###
############################
class SparseToDenseDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_data, labels):
        self.data = sparse_data
        self.labels = labels
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        # 只在需要时转换单个样本为密集格式
        dense_sample = torch.FloatTensor(self.data[idx].toarray().squeeze())
        label_tensor = torch.LongTensor([self.labels[idx]]).squeeze()
        return dense_sample.unsqueeze(0), label_tensor
##evaluate model
## evaluate the model with test set
def evaluate_model(model, test_loader, device, OutputPath,label_names=None):
    """
    评估模型在测试集上的性能
    Args:
        model: 训练好的模型
        test_loader: 测试集DataLoader
        device: 计算设备 (cuda/cpu)
        label_names: 可选的类别名称列表
    Returns:
        metrics_dict: 包含各项指标的字典
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            print('len of all_labels:', len(all_labels))

    # 计算分类报告
    report = classification_report(
        all_labels, 
        all_preds,
        target_names=label_names,
        output_dict=True,
        digits=4
    )
    print("Classification Report:finished")
    # 直接转换为DataFrame
    report_df = pd.DataFrame(report).transpose()  # 转置使指标为列
    # 保存为CSV
    report_df.to_csv(f'{OutputPath}/classification_report.csv', float_format='%.4f')
    print("Classification Report:finished")
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm,
    index=[f"True_{i}" for i in range(cm.shape[0])],
    columns=[f"Pred_{i}" for i in range(cm.shape[1])])
    # 保存为CSV
    cm_df.to_csv(f'{OutputPath}/confusion_matrix.csv')
    print("Confusion Matrix:finished")
    # 可视化混淆矩阵
    plt.figure(figsize=(28, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, 
                yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{OutputPath}/confusion_matrix.png')
    plt.close()


## config save and load
def save_config(config, filename):
    """安全保存配置"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def load_config(filename):
    """安全加载配置"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


##search space
def HyperparameterTune(anndata_obj,y,train_indices, dev_indices,test_indices, config={},initcpu=1,initgpu=1,onetrialcpu=0.5,onetrialgpu=0.5,evaluate_classifier=False):
    """
    use ray.tune to do hyperparameter tuning
    Parameters:
    ----------------------------------------------------
    config: a dictionary containing all the parameters. For example:
    config={'classifier':CellTypeClassifierLinear,
        'TrainingFunc':Train_CellType_Annotation,
        'label_names':label_encoder.classes_,
        'metriclst':["train_loss", "dev_f1", "best_f1", "epoch"],
        'metric_standard':'dev_f1',
        'metric_mode':'max',
        'num_epochs':30,
        'num_epochs_atleast':10,
        'totaltrials':30,
        'learning_rate':tune.loguniform(1e-5, 1e-3),
        'batch_size':tune.choice([500,1000]),
        'hidden_neuron':tune.choice([400, 800,1000]),
        "use_subset":False,
        'outputPath':'./CellTypeAnnotation/Linear',
        'filename':"scCellTypeAnnotation_tune",}  # 超参数配置
    anndata_obj: anndata object containing the data.
    """
    ray.init(num_cpus=initcpu, num_gpus=initgpu)  # 根据实际资源调整
    start_time = time.time()  # 记录开始时间
    # 将大型数据存入Ray对象存储
    config['train_data_ref'] = ray.put(SparseToDenseDataset(anndata_obj.X[train_indices], y[train_indices]))
    config['dev_data_ref'] = ray.put(SparseToDenseDataset(anndata_obj.X[dev_indices], y[dev_indices]))

    # 配置调度器（ASHA算法）
    scheduler = ASHAScheduler(
        max_t=config['num_epochs'],
        grace_period=config['num_epochs_atleast'],    # 至少运行10个epoch
        reduction_factor=2, # 中期淘汰表现差的试验
        metric=config['metric_standard'],
        mode=config['metric_mode']
    )
    
    # 配置结果报告
    reporter = CLIReporter(metric_columns=config['metriclst'])
    tuner = tune.Tuner(
        tune.with_resources(
        config['TrainingFunc'],
        resources={"cpu": onetrialcpu, "gpu": onetrialgpu}  # assign resources for each trial (1 CPU and 0.2 GPU)
        ),
        tune_config=tune.TuneConfig(
            num_samples=config['totaltrials'], ##total trials
            scheduler=scheduler,
        ),
        run_config=tune.RunConfig(
            storage_path=config['outputPath'],
            name=config['filename'],
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=2,#keep the best checkpoint for each trial
                checkpoint_score_attribute=config['metric_standard'],
                checkpoint_score_order=config['metric_mode']
            )
        ),
        param_space=config
    )
    
    results = tuner.fit()
    # 方法1：直接获取最佳结果（最简洁）
    best_result = results.get_best_result(metric=config['metric_standard'], mode=config['metric_mode'])
    print(f"Best trial achieved F1 Score: {best_result.metrics[config['metric_standard']]:.4f}")
    # 从best_result中可以直接获取配置信息
    best_config = best_result.config
    print(f"Best hyperparameters: {best_config}")
    
    if best_result.checkpoint:
        # 从检查点加载数据
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            # 加载模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 获取classifier的构造函数参数
            classifier_init_signature = inspect.signature(config['classifier'].__init__)
            valid_params = classifier_init_signature.parameters.keys()
            # 过滤出有效的参数
            model_params = {k: v for k, v in best_config.items() 
                       if k in valid_params and k != 'self' and k != 'args' and k != 'kwargs'}
            save_config(model_params, f"{config['outputPath']}/best_model_hyperparameters.json")
            best_model = config['classifier'](**model_params)
            
            # 加载模型权重
            model_path = os.path.join(checkpoint_dir, "model.pt")
            best_model.load_state_dict(torch.load(model_path, map_location=device))
            best_model.to(device)
            torch.save(best_model.state_dict(), f"{config['outputPath']}/best_model.pth")
            os.makedirs(f"{config['outputPath']}/Evaluation", exist_ok=True)
            if evaluate_classifier==True:
                evaluate_model(best_model, DataLoader(SparseToDenseDataset(anndata_obj.X[test_indices], y[test_indices]), batch_size=5000), device, label_names=config['label_names'],OutputPath=f"{config['outputPath']}/Evaluation")
    
    ray.shutdown()
    end_time = time.time()
    elapsed = end_time - start_time  # calculate elapsed time in seconds
    elapsed_minutes = elapsed/ 60  # transform to minutes
    print(f"it takes {elapsed_minutes:.2f} minutes")
