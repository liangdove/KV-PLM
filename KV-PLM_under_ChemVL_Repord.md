# Reproduce KV-PLM Project Under ChemVL Data Splitting and Evaluation Framework

## System Environment
#### Image Version
- PyTorch: 1.6.0
- Python: 3.8 (Ubuntu 18.04)
- CUDA: 10.1

#### Hardware Configuration
- GPU: 1 × RTX 2080 Ti (11GB)
- CPU: 12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz

## 1. Environment Setup
Follow the official environment configuration of ChemVL to install all required dependencies. For detailed steps, refer to `README.md`.

## 2. Execute chemvl_to_molecule.py
This script parses ChemVL CSV files, generates `sm_*.npy`, `lab_*.npy`, `text_*.txt` files required by KV-PLM, and outputs split files named `split_<TASK>_<mode>_{train,valid,test}.npy`.

Run commands under the project root directory:
1. Generate KV-PLM formatted MoleculeNet files with scaffold split
```bash
python KV-PLM/tools/chemvl_to_moleculenet.py --task HIV --split_mode scaffold
```

2. Generate files with random-scaffold split
```bash
python KV-PLM/tools/chemvl_to_moleculenet.py --task HIV --split_mode random_scaffold
```

### Notes
1. `chemvl_root`: Path to ChemVL dataset folder
    `/xxx/chemvl-data/finetuning_datasets/MPP/classification`
2. `chemvl_private_root`: Root path of ChemVL project
    `/xxx/ChemVL-private`

## 3. Execute run_molecule.py
This script adopts ChemVL standard MoleculeNet data splitting strategy and evaluation pipeline.

Run commands inside the KV-PLM folder:
```bash
cd KV-PLM
mkdir finetune_save
```

```bash
python run_molecule.py \
--task HIV \
--split_idx_dir MoleculeNet \
--split_mode random_scaffold \
--output finetune_save/KV-PLM_HIV_under_ChemVL_random_scaffold_chemvl_split_metrics.pt \
--init_checkpoint 'save_model/ckpt_KV_1.pt' \
```

## 4. Batch Execution Scripts
```bash
cd KV-PLM
sh run_all_classfifcation.sh
sh run_all_regression.sh
```
