# 这一版代码能够保证分类任务正确运行
import argparse
import pickle
import os
import random
import json
import time
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler 
from tqdm import tqdm, trange
import modeling 
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from sklearn.metrics import f1_score,roc_auc_score
from chainer_chemistry.dataset.splitters.scaffold_splitter import ScaffoldSplitter
from transformers import AutoTokenizer,AutoModel,BertModel,BertForPreTraining,BertConfig
from smtokenization import SmilesTokenizer
from rdkit import Chem

class OldModel(nn.Module):
    def __init__(self, pt_model):
        super(OldModel, self).__init__()
        self.ptmodel = pt_model
        self.emb = nn.Embedding(390, 768)

    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        embs = self.ptmodel.bert.embeddings.word_embeddings(input_ids)
        msk = torch.where(input_ids>=30700)
        for k in range(msk[0].shape[0]):
            i = msk[0][k].item()
            j = msk[1][k].item()
            embs[i,j] = self.emb(input_ids[i,j]-30700)
        '''
        msk = (input_ids>=30700)
        embs = self.emb((input_ids-30700)*msk)
        return self.ptmodel.bert(inputs_embeds=embs, attention_mask=attention_mask, token_type_ids=token_type_ids)

class BigModel(nn.Module):
    def __init__(self, bert_model, config, multi):
        super(BigModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if multi==0:
            self.classifier = nn.Linear(config.hidden_size, 2)
        else:
            self.classifier = []
            for i in range(multi):
                self.classifier.append(nn.Linear(config.hidden_size, 2))
            self.classifier = nn.ModuleList(self.classifier)
        self.multi = multi

    def forward(self, tokens, token_type_ids, attention_mask):
        pooled = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)['pooler_output']
        encoded = self.dropout(pooled)
        if self.multi==0:
            return self.classifier(encoded)
        return [ self.classifier[i](encoded) for i in range(self.multi) ]


class Mole_dataset(Dataset):
    def __init__(self, pth_data, pth_lab, pth_text, seq, tok, rx):
        self.data = np.load(pth_data, allow_pickle=True)
        self.lab = np.load(pth_lab)
        self.seq = seq
        self.tok = tok
        if tok:
            f = open(pth_text, 'r')
            self.data = f.readlines()
            self.tokenizer = AutoTokenizer.from_pretrained('/root/KV-PLM/scibert')
            if rx:
                self.tokenizer = SmilesTokenizer.from_pretrained('rxnfp/rxnfp/models/transformers/bert_mlm_1k_tpl')

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        if self.tok:
            index = self.seq[index]
            lab = self.lab[index]
            token = self.tokenizer.encode(self.data[index].strip('\n'))
            tok = np.zeros(64)
            att = np.zeros(64)
            tok[:min(64, len(token))] = token[:min(64, len(token))]
            att[:min(64, len(token))] = 1
            return torch.tensor(tok.copy()).long(), torch.tensor(lab.copy()).long(),torch.tensor(att.copy()).long()
        prop = random.randint(0,9)
        sq = self.seq[index]
        dat = np.zeros(32)
        sub = [102] +[ i+30700 for i in self.data[sq] ] + [103]
        dat[:min(32, len(sub))] = sub[:min(32, len(sub))]
        lab = self.lab[sq]
        att = np.zeros(32)
        att[:min(32, len(sub))] = np.ones(min(32, len(sub)))
        
        return torch.tensor(dat.copy()).long(), torch.tensor(lab.copy()).long(),torch.tensor(att.copy()).long()
        
def prepare_model_and_optimizer(args, device):
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
        
    bert_model0 = BertForPreTraining.from_pretrained('/root/KV-PLM/scibert')
    bert_model = OldModel(bert_model0)
    
    if args.init_checkpoint=='BERT':
        con = BertConfig(vocab_size=31090,)
        bert_model = BertModel(con)
        args.tok = 1
        model = BigModel(bert_model, config, args.multi)
    elif args.init_checkpoint=='rxnfp':
        bert_model =  BertModel.from_pretrained('rxnfp/transformers/bert_mlm_1k_tpl')
        args.pth_data += 'rxnfp/'
        config.hidden = 256
        args.tok = 1
        model = BigModel(bert_model, config, args.multi)
        args.rx = 1
    elif args.init_checkpoint is None:
        args.tok = 1
        model = BigModel(bert_model0.bert, config, args.multi)
    else:
        pt = torch.load(args.init_checkpoint)
        if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in pt:
            pretrained_dict = {k[7:]: v for k, v in pt.items()}
            args.tok = 0
            bert_model.load_state_dict(pretrained_dict, strict=False)
            model = BigModel(bert_model, config, args.multi)
        elif 'bert.embeddings.word_embeddings.weight' in pt:
            #pretrained_dict = {k[5:]: v for k, v in pt.items()}
            args.tok = 1
            bert_model0.load_state_dict(pt, strict=True)
            model = BigModel(bert_model0.bert, config, args.multi)
    #model = torch.nn.DataParallel(model)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer = BertAdam(
            optimizer_grouped_parameters,
            weight_decay=args.weight_decay,
            lr=args.lr,
            warmup=args.warmup,
            t_total=args.total_steps,
            )
    return model,optimizer


def _add_chemvl_private_to_path(chemvl_private_root: str) -> None:
    chemvl_private_root = os.path.abspath(chemvl_private_root)
    if chemvl_private_root not in sys.path:
        sys.path.insert(0, chemvl_private_root)


def _detect_task_type(task_name):
    """Detect task type (classification or regression) based on task name."""
    # All MoleculeNet tasks in KV-PLM are classification by default
    # Only specific datasets are regression tasks
    regression_tasks = ["esol", "freesolv", "lipophilicity", "qm7"]
    task_lower = task_name.strip().lower()
    if task_lower in regression_tasks:
        return "regression"
    return "classification"


def _select_primary_metric(metrics_dict, task_type="classification"):
    if task_type == "regression":
        # For regression, prefer RMSE, then MAE, then R2
        for key in ["RMSE", "MAE", "R2"]:
            if key in metrics_dict:
                return float(metrics_dict[key])
    else:
        # For classification, prefer ROCAUC
        for key in ["ROCAUC", "RMSE", "MAE", "R2"]:
            if key in metrics_dict:
                return float(metrics_dict[key])
    raise KeyError("No supported primary metric found in metrics dict.")

def Eval(model, dataloader, multi, eval_metric_fn, eval_metric_multitask_fn, task_type="classification"):
    model.eval()
    with torch.no_grad():
        acc = 0
        allcnt = 0
        y_true_list = []
        y_pro_list = []
        for batch in tqdm(dataloader):
            (tok, lab, att) = batch
            typ = torch.zeros(tok.shape).long().cuda()
            logits = model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())
            
            if task_type == "regression":
                # Regression: logits are directly the predictions
                if multi > 0:
                    y_true_list.append(lab.cpu().numpy())
                    y_pred_batch = logits
                    if isinstance(logits, list):
                        y_pred_batch = torch.stack(logits, dim=1)
                    y_pro_list.append(y_pred_batch.cpu().numpy())
                else:
                    y_true_list.append(lab.cpu().numpy())
                    y_pro_list.append(logits.cpu().numpy())
            else:
                # Classification: apply sigmoid and convert to probabilities
                if multi > 0:
                    y_true_list.append(lab.cpu().numpy())
                    y_score = torch.nn.Sigmoid()(logits[0][:,1]-logits[0][:,0]).unsqueeze(0)
                    for i in range(1, len(logits)):
                        y_score = torch.cat(
                            (y_score, torch.nn.Sigmoid()(logits[i][:,1]-logits[i][:,0]).unsqueeze(0)),
                            axis=0,
                        )
                    y_pro_list.append(y_score.transpose(0, 1).cpu().numpy())
                else:
                    y_true_list.append(lab.cpu().numpy())
                    y_score = torch.nn.Sigmoid()(logits[:,1]-logits[:,0]).unsqueeze(1)
                    y_pro_list.append(y_score.cpu().numpy())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pro = np.concatenate(y_pro_list, axis=0)
    
    if task_type == "regression":
        # For regression: y_pro contains predictions, no thresholding needed
        y_pred = y_pro
        if multi == 0:
            return eval_metric_fn(y_true.squeeze(), y_pred.squeeze())
        return eval_metric_multitask_fn(y_true, y_pred, num_tasks=y_true.shape[1])
    else:
        # For classification: apply threshold
        y_pred = (y_pro >= 0.5).astype(int)
        if multi == 0:
            return eval_metric_fn(y_true.squeeze(), y_pred.squeeze(), y_pro.squeeze(), empty=-1)
        return eval_metric_multitask_fn(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
    
def main(args):
    device = torch.device('cuda')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Resolve chemvl_private_root with fallbacks if provided path doesn't exist
    candidates = []
    if args.chemvl_private_root:
        candidates.append(os.path.abspath(args.chemvl_private_root))
        candidates.append(os.path.abspath(os.path.join(os.getcwd(), args.chemvl_private_root)))
    # sibling path relative to KV-PLM repo dir
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ChemVL-private')))
    # parent of workspace
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ChemVL-private')))
    # common absolute path used in this environment
    candidates.append('/home/administrator/code_liangdove/ChemVL-private')

    chemvl_root_resolved = None
    for c in candidates:
        if c and os.path.isdir(c):
            chemvl_root_resolved = c
            break

    if chemvl_root_resolved is None:
        print("Warning: could not locate ChemVL-private directory from candidates:\n" + "\n".join([str(x) for x in candidates]))
        print("Please pass a valid --chemvl_private_root path (absolute or relative).\nFalling back to args.chemvl_private_root as given.")
        chemvl_root_resolved = args.chemvl_private_root

    print(f"Using ChemVL-private root: {chemvl_root_resolved}")
    _add_chemvl_private_to_path(chemvl_root_resolved)
    from models.evaluate import metric as chemvl_metric
    from models.evaluate import metric_multitask as chemvl_metric_multitask
    from models.evaluate import metric_reg as chemvl_metric_reg
    from models.evaluate import metric_reg_multitask as chemvl_metric_reg_multitask

    def task_to_kv_name(task):
        lower = task.strip().lower()
        if lower in ['tox21', 'sider']:
            return lower
        return task.strip().upper()

    def load_external_split_indices(split_dir, task, split_mode):
        kv_name = task_to_kv_name(task)
        mode = split_mode.strip().lower()
        train_path = os.path.join(split_dir, f"split_{kv_name}_{mode}_train.npy")
        valid_path = os.path.join(split_dir, f"split_{kv_name}_{mode}_valid.npy")
        test_path = os.path.join(split_dir, f"split_{kv_name}_{mode}_test.npy")
        if not (os.path.isfile(train_path) and os.path.isfile(valid_path) and os.path.isfile(test_path)):
            raise FileNotFoundError(
                f"Split indices not found: {train_path}, {valid_path}, {test_path}"
            )
        return [
            np.load(train_path),
            np.load(valid_path),
            np.load(test_path),
        ]

    def filter_valid_smiles(sm_list):
        valid_idx = []
        valid_smiles = []
        for i, smi in enumerate(sm_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            valid_idx.append(i)
            valid_smiles.append(smi)
        if len(valid_idx) != len(sm_list):
            print('Filtered invalid SMILES:', len(sm_list) - len(valid_idx))
        return np.array(valid_idx), np.array(valid_smiles)

    use_external_split = args.split_idx_dir is not None
    task_type = _detect_task_type(args.task)
    print(f"Detected task type for {args.task}: {task_type}")

    if args.task=='tox21':
        args.sm_pth = args.sm_pth + 'tox21.npy'
        args.pth_lab = args.pth_lab + 'tox21.npy'
        args.pth_data = args.pth_data + 'tox21.npy'
        args.pth_text = args.pth_text + 'tox21.txt'
        args.total_steps = 1200
        args.multi = 12
        sm_list = np.load(args.sm_pth, allow_pickle=True)
        if use_external_split:
            scaf = load_external_split_indices(args.split_idx_dir, args.task, args.split_mode)
            print('Loaded external split indices for tox21.')
        else:
            seq = np.arange(len(sm_list))
            np.random.shuffle(seq)
            scaf = []
            k = int(len(seq)/10)
            scaf.append(seq[:8*k])
            scaf.append(seq[8*k:9*k])
            scaf.append(seq[9*k:])
    elif args.task=='HIV':
        args.sm_pth = args.sm_pth + 'HIV.npy'
        args.pth_lab = args.pth_lab + 'HIV.npy'
        args.pth_data = args.pth_data + 'HIV.npy'
        args.pth_text = args.pth_text + 'HIV.txt'
        args.total_steps = 6000
        args.multi = 0
        sm_list = np.load(args.sm_pth, allow_pickle=True)
        if use_external_split:
            scaf = load_external_split_indices(args.split_idx_dir, args.task, args.split_mode)
            print('Loaded external split indices for HIV.')
        else:
            seq, sm_list = filter_valid_smiles(sm_list)
            sp = ScaffoldSplitter()
            scaf = sp.train_valid_test_split(dataset=seq, smiles_list=sm_list, frac_train=0.8,
                                   frac_valid=0.1, frac_test=0.1,include_chirality=False)
    elif args.task=='BBBP':
        args.sm_pth = args.sm_pth + 'BBBP.npy'
        args.pth_lab = args.pth_lab + 'BBBP.npy'
        args.pth_data = args.pth_data + 'BBBP.npy'
        args.pth_text = args.pth_text + 'BBBP.txt'
        args.total_steps = 300
        args.multi = 0
        sm_list = np.load(args.sm_pth, allow_pickle=True)
        if use_external_split:
            scaf = load_external_split_indices(args.split_idx_dir, args.task, args.split_mode)
            print('Loaded external split indices for BBBP.')
        else:
            seq, sm_list = filter_valid_smiles(sm_list)
            sp = ScaffoldSplitter()
            scaf = sp.train_valid_test_split(dataset=seq, smiles_list=sm_list, frac_train=0.8,
                                   frac_valid=0.1, frac_test=0.1,include_chirality=False)
    elif args.task=='sider':
        args.sm_pth = args.sm_pth + 'sider.npy'
        args.pth_lab = args.pth_lab + 'sider.npy'
        args.pth_data = args.pth_data + 'sider.npy'
        args.pth_text = args.pth_text + 'sider.txt'
        args.total_steps = 200
        args.multi = 27
        sm_list = np.load(args.sm_pth, allow_pickle=True)
        if use_external_split:
            scaf = load_external_split_indices(args.split_idx_dir, args.task, args.split_mode)
            print('Loaded external split indices for sider.')
        else:
            seq = np.arange(len(sm_list))
            np.random.shuffle(seq)
            scaf = []
            k = int(len(seq)/10)
            scaf.append(seq[:8*k])
            scaf.append(seq[8*k:9*k])
            scaf.append(seq[9*k:])
    else:
        # Generic fallback for tasks not explicitly enumerated above (e.g., BACE, CLINTOX, ESOL, FREESOLV, LIPOPHILICITY, QM7)
        kv = task_to_kv_name(args.task)
        args.sm_pth = args.sm_pth + f'{kv}.npy'
        args.pth_lab = args.pth_lab + f'{kv}.npy'
        args.pth_data = args.pth_data + f'{kv}.npy'
        args.pth_text = args.pth_text + f'{kv}.txt'
        args.total_steps = getattr(args, 'total_steps', 1200)
        
        sm_list = np.load(args.sm_pth, allow_pickle=True)
        # Auto-detect multi-task by checking label shape
        lab_data = np.load(args.pth_lab)
        if len(lab_data.shape) > 1 and lab_data.shape[1] > 1:
            # Multi-task: labels shape is (N, num_tasks)
            args.multi = lab_data.shape[1]
        else:
            # Single-task: labels shape is (N,)
            args.multi = 0
        print(f"Auto-detected args.multi={args.multi} for task {args.task}")
        
        if use_external_split:
            scaf = load_external_split_indices(args.split_idx_dir, args.task, args.split_mode)
            print(f'Loaded external split indices for {args.task}.')
        else:
            if task_type == "regression":
                seq, sm_list = filter_valid_smiles(sm_list)
                sp = ScaffoldSplitter()
                scaf = sp.train_valid_test_split(dataset=seq, smiles_list=sm_list, frac_train=0.8,
                                       frac_valid=0.1, frac_test=0.1,include_chirality=False)
            else:
                seq = np.arange(len(sm_list))
                np.random.shuffle(seq)
                scaf = []
                k = int(len(seq)/10)
                scaf.append(seq[:8*k])
                scaf.append(seq[8*k:9*k])
                scaf.append(seq[9*k:])

    model, optimizer = prepare_model_and_optimizer(args, device)
    
    TrainSet = Mole_dataset(args.pth_data, args.pth_lab, args.pth_text, scaf[0], args.tok, args.rx)
    DevSet = Mole_dataset(args.pth_data, args.pth_lab, args.pth_text, scaf[1], args.tok, args.rx)
    TestSet = Mole_dataset(args.pth_data, args.pth_lab, args.pth_text, scaf[2], args.tok, args.rx)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=True, drop_last=False)
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=True, drop_last=False)
    test_dataloader = DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=True, drop_last=False)
    loss_func = torch.nn.CrossEntropyLoss()
    if task_type == "regression":
        loss_func = torch.nn.MSELoss()
    global_step = 0
    tag = True
    best_acc = np.inf if task_type == "regression" else 0
    for epoch in range(args.epoch):
        if tag==False:
            break
        
        if task_type == "regression":
            dev_metrics = Eval(model, dev_dataloader, args.multi, chemvl_metric_reg, chemvl_metric_reg_multitask, task_type=task_type)
        else:
            dev_metrics = Eval(model, dev_dataloader, args.multi, chemvl_metric, chemvl_metric_multitask, task_type=task_type)
        dev_score = _select_primary_metric(dev_metrics, task_type=task_type)
        print('Epoch:', epoch, ', DevMetric:', dev_metrics)
        # For regression: lower is better (RMSE/MAE); for classification: higher is better (ROCAUC)
        should_update = (task_type == "regression" and dev_score < best_acc) or (task_type == "classification" and dev_score > best_acc)
        if should_update:
            best_acc = dev_score
            torch.save(model.state_dict(), args.output)
            print('Save checkpoint ', global_step)
        
        acc = 0
        allcnt = 0
        sumloss = 0
        y_true = None
        y_score = None
        model.train()
        
        for idx,batch in enumerate(tqdm(train_dataloader)):
            (tok, lab, att) = batch
            typ = torch.zeros(tok.shape).long().cuda()
            logits = model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())
            
            if task_type == "regression":
                # Regression task: logits are predictions, apply MSE loss
                if args.multi == 0:
                    loss = loss_func(logits.view(-1), lab.cuda().view(-1))
                else:
                    loss = torch.tensor(0.0).cuda()
                    for i in range(len(logits)):
                        for j in range(lab.shape[0]):
                            if lab[j, i].item() == -1:
                                continue
                            loss += loss_func(
                                logits[i][j].unsqueeze(0),
                                lab[j, i].cuda().unsqueeze(0),
                            )
            else:
                # Classification task: apply CrossEntropy loss
                if args.multi == 0:
                    loss = loss_func(logits.view(-1, 2),
                            lab.cuda().view(-1),
                            )
                else:
                    loss = torch.tensor(0.0).cuda()
                    for i in range(len(logits)):
                        for j in range(lab.shape[0]):
                            if lab[j, i].item() == -1:
                                continue
                            loss += loss_func(
                                (logits[i].view(-1, args.num_labels)[j]).unsqueeze(0),
                                lab[j, i].cuda().view(-1),
                                )
            
            allcnt += tok.shape[0]
            sumloss += loss.item()
            loss.backward()
            if idx%2==1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if global_step>args.total_steps:
                tag = False
                break
        optimizer.step()
        optimizer.zero_grad()
        print('Epoch:', epoch, ', Loss:', sumloss/allcnt)

    if task_type == "regression":
        dev_metrics = Eval(model, dev_dataloader, args.multi, chemvl_metric_reg, chemvl_metric_reg_multitask, task_type=task_type)
    else:
        dev_metrics = Eval(model, dev_dataloader, args.multi, chemvl_metric, chemvl_metric_multitask, task_type=task_type)
    dev_score = _select_primary_metric(dev_metrics, task_type=task_type)
    print('Epoch:', args.epoch, ', DevMetric:', dev_metrics)
    if (task_type == "regression" and dev_score < best_acc) or (task_type == "classification" and dev_score > best_acc):
        best_acc = dev_score
        torch.save(model.state_dict(), args.output)
        print('Save checkpoint ', global_step)
    model.load_state_dict(torch.load(args.output))
    if task_type == "regression":
        test_metrics = Eval(model, test_dataloader, args.multi, chemvl_metric_reg, chemvl_metric_reg_multitask, task_type=task_type)
    else:
        test_metrics = Eval(model, test_dataloader, args.multi, chemvl_metric, chemvl_metric_multitask, task_type=task_type)
    print('Test Metric:', test_metrics)

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--config_file", default='bert_base_config.json', type=str,)
    parser.add_argument("--num_labels", default=2, type=int,)
    parser.add_argument("--init_checkpoint", default=None, type=str,)
    parser.add_argument("--task", default='tox21', type=str,)
    parser.add_argument("--multi", default=1, type=int,)
    parser.add_argument("--tok", default=0, type=int,)
    parser.add_argument("--rx", default=0, type=int,)
    parser.add_argument("--sm_pth", default='MoleculeNet/sm_', type=str,)
    parser.add_argument("--resume", default=-1, type=int,)
    parser.add_argument("--weight_decay", default=0, type=float,)
    parser.add_argument("--lr", default=5e-6, type=float,)
    parser.add_argument("--warmup", default=0.2, type=float,)
    parser.add_argument("--total_steps", default=1200, type=int,)
    parser.add_argument("--pth_data", default='MoleculeNet/sub_', type=str,)
    parser.add_argument("--pth_lab", default='MoleculeNet/lab_', type=str,)
    parser.add_argument("--pth_text", default='MoleculeNet/text_', type=str,)
    parser.add_argument("--batch_size", default=64, type=int,)
    parser.add_argument("--epoch", default=20, type=int,)
    parser.add_argument("--seed", default=1011, type=int,)
    parser.add_argument("--output", default='finetune_save/ckpt_test1.pt', type=str,)
    parser.add_argument("--split_idx_dir", default=None, type=str,)
    parser.add_argument("--split_mode", default="scaffold", choices=["scaffold", "random_scaffold"], type=str,)
    parser.add_argument(
        "--chemvl_private_root",
        default="ChemVL-private",
        type=str,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
