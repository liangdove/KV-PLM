#!/bin/bash

echo "======== scaffold划分 ========"
echo "======== 开始执行 BBBP 任务 ========"
python run_molecule.py --task BBBP --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_BBBP_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > BBBP_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1

echo "======== BBBP 任务完成，开始执行 sider 任务 ========"
python run_molecule.py --task sider --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_sider_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > sider_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1

echo "======== sider 任务完成，开始执行 tox21 任务 ========"
python run_molecule.py --task tox21 --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_tox21_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > tox21_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1

echo "======== tox21 任务完成，开始执行 HIV 任务 ========"
python run_molecule.py --task HIV --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_HIV_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > HIV_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1

echo "======== HIV 任务完成，开始执行 BACE 任务 ========"
python run_molecule.py --task BACE --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_BACE_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > BACE_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1

echo "======== BACE 任务完成，开始执行 clintox 任务 ========"
python run_molecule.py --task CLINTOX --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_CLINTOX_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > CLINTOX_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1



echo "======== random scaffold划分 ========"
echo "======== 开始执行 HIV 任务 ========"
python run_molecule.py --task HIV --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_HIV_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > HIV_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1

echo "======== HIV 任务完成，开始执行 BBBP 任务 ========"
python run_molecule.py --task BBBP --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_BBBP_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > BBBP_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1

echo "======== BBBP 任务完成，开始执行 sider 任务 ========"
python run_molecule.py --task sider --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_sider_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > sider_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1

echo "======== sider 任务完成，开始执行 tox21 任务 ========"
python run_molecule.py --task tox21 --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_tox21_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > tox21_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1

echo "======== HIV 任务完成，开始执行 BACE 任务 ========"
python run_molecule.py --task BACE --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_BACE_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > BACE_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1

echo "======== BACE 任务完成，开始执行 CLINTOX 任务 ========"
python run_molecule.py --task CLINTOX --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_CLINTOX_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > CLINTOX_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1


echo "======== 所有任务执行完毕！ ========"
