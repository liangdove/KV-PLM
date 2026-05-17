#!/bin/bash

echo "======== scaffold划分 ========"
echo "======== 开始执行 ESOL 任务 ========"
python run_molecule.py --task ESOL --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_ESOL_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > ESOL_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1

echo "======== ESOL 任务完成，开始执行 FREESOLV 任务 ========"
python run_molecule.py --task FREESOLV --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_FREESOLV_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > FREESOLV_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1

echo "======== FREESOLV 任务完成，开始执行 LIPOPHILICITY 任务 ========"
python run_molecule.py --task LIPOPHILICITY --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_LIPOPHILICITY_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > LIPOPHILICITY_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1

echo "======== LIPOPHILICITY 任务完成，开始执行 QM7 任务 ========"
python run_molecule.py --task QM7 --split_idx_dir MoleculeNet --split_mode scaffold --output finetune_save/KV-PLM_QM7_under_ChemVL_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > QM7_under_ChemVL_scaffold_chemvl_split_metrics.log 2>&1



echo "======== random scaffold划分 ========"
echo "======== 开始执行 ESOL 任务 ========"
python run_molecule.py --task ESOL --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_ESOL_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > ESOL_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1

echo "======== ESOL 任务完成，开始执行 FREESOLV 任务 ========"
python run_molecule.py --task FREESOLV --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_FREESOLV_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > FREESOLV_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1

echo "======== FREESOLV 任务完成，开始执行 LIPOPHILICITY 任务 ========"
python run_molecule.py --task LIPOPHILICITY --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_LIPOPHILICITY_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > LIPOPHILICITY_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1

echo "======== LIPOPHILICITY 任务完成，开始执行 QM7 任务 ========"
python run_molecule.py --task QM7 --split_idx_dir MoleculeNet --split_mode random_scaffold --output finetune_save/KV-PLM_QM7_under_ChemVL_random_scaffold_chemvl_split_metrics.pt --init_checkpoint 'save_model/ckpt_KV_1.pt' > QM7_under_ChemVL_random_scaffold_chemvl_split_metrics.log 2>&1

echo "======== 所有任务执行完毕！ ========"
