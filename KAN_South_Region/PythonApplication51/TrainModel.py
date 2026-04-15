#-*- coding : utf-32 -*-
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics

import sklearn.preprocessing
import sklearn.metrics

import matplotlib.pyplot as plt

import tqdm
import MakeDataset

import GetModel
import DrawLog
import CONFIG
import PATH

def train_func(loader, model, optimizer, loss_func, metrics_func, device): 
    model.train() 
    
    batch_loss_list, batch_metrics_list = [], []
    
    # Mini-batch
    pbar = tqdm.tqdm(loader)
    for batch in pbar:
        
        inputs_pos, inputs_fac, targets = batch

        inputs_pos = inputs_pos.to(device)
        inputs_fac = inputs_fac.to(device)
        targets = targets.to(device)

        outputs = model(inputs_pos, inputs_fac)  

        loss = loss_func(outputs, targets)  
        loss = loss.mean()
        
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        
        metrics = metrics_func(outputs, targets) 
        
        batch_loss_list.append(loss.item())
        batch_metrics_list.append(metrics.item())
        pbar.set_postfix(loss='{:.4f}'.format(loss.item()), metrics='{:.4f}'.format(metrics.item()))  # 进度条右边显示信息
    
    return np.mean(batch_loss_list), np.mean(batch_metrics_list), np.std(batch_loss_list), np.std(batch_metrics_list)

@torch.no_grad
def val_func(loader, model, loss_func, metrics_func, device):    
    model.eval() 

    batch_loss_list, batch_metrics_list = [], []
    
    pbar = tqdm.tqdm(loader)
    for batch in pbar:
        
        inputs_pos, inputs_fac, targets = batch

        inputs_pos = inputs_pos.to(device)
        inputs_fac = inputs_fac.to(device)
        targets = targets.to(device)

        outputs = model(inputs_pos, inputs_fac) 

        loss = loss_func(outputs, targets)  
        loss = loss.mean()

        metrics = metrics_func(outputs, targets) 

        batch_loss_list.append(loss.item())
        batch_metrics_list.append(metrics.item())
        pbar.set_postfix(loss='{:.4f}'.format(loss.item()), metrics='{:.4f}'.format(metrics.item()))  # 进度条右边显示信息
    
    return np.mean(batch_loss_list), np.mean(batch_metrics_list), np.std(batch_loss_list), np.std(batch_metrics_list)

@torch.no_grad
def test_func(loader, model, device):
    model.eval() 

   
    all_probs = []   #  [p0, p1, ..., p6]
    all_targets = [] # （0-6）
    
    pbar = tqdm.tqdm(loader)
    for batch in pbar:
        
        inputs_pos, inputs_fac, targets = batch

        inputs_pos = inputs_pos.to(device)
        inputs_fac = inputs_fac.to(device)
        #targets = targets.to(device)

        outputs = model(inputs_pos, inputs_fac)  
        probs = F.softmax(outputs, dim=1)     

        all_probs.append(probs.cpu())         
        all_targets.append(targets)     
    else:
       
        all_probs = torch.cat(all_probs, dim=0).numpy()         # shape: [N, 7]
        all_targets = torch.cat(all_targets, dim=0).numpy()     # shape: [N]
        pass

    all_pred_labels = all_probs.argmax(axis=1)

    all_targets_onehot = sklearn.preprocessing.label_binarize(
        all_targets,
        classes=list(range(CONFIG.NUM_CLASSES)))  # shape: [N, 7]

    oacc_score = sklearn.metrics.accuracy_score(all_targets, all_pred_labels)
    bacc_score = sklearn.metrics.balanced_accuracy_score(all_targets, all_pred_labels)
    f1_macro = sklearn.metrics.f1_score(all_targets, all_pred_labels, average='macro')
    f1_micro = sklearn.metrics.f1_score(all_targets, all_pred_labels, average='micro')
    f1_weighted = sklearn.metrics.f1_score(all_targets, all_pred_labels, average='weighted')
    roc_auc_macro = sklearn.metrics.roc_auc_score(all_targets_onehot, all_probs, average='macro', multi_class='ovr')
    roc_auc_micro = sklearn.metrics.roc_auc_score(all_targets_onehot, all_probs, average='micro', multi_class='ovr')
    pr_auc_macro = sklearn.metrics.average_precision_score(all_targets_onehot, all_probs, average='macro')
    pr_auc_micro = sklearn.metrics.average_precision_score(all_targets_onehot, all_probs, average='micro')

    print("Overall Acc:", oacc_score)
    print("Balanced Acc:", bacc_score)
    print("Macro F1-Score", f1_macro)
    print("Micro F1-Score", f1_micro)
    print("Weighted F1-Score", f1_weighted)
    print("ROC AUC (macro):", roc_auc_macro)
    print("ROC AUC (micro):", roc_auc_micro)
    print("PR AUC (macro):", pr_auc_macro)
    print("PR AUC (micro):", pr_auc_micro)

    print(sklearn.metrics.classification_report(all_targets, all_pred_labels))
    #print(sklearn.metrics.confusion_matrix(all_targets, all_pred_labels))
    conf_mat = sklearn.metrics.confusion_matrix(all_targets, all_pred_labels)  # shape: [num_classes, num_classes]
    cm_percent = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] * 100
    print("Confusion Matrix (raw counts):\n", conf_mat)
    print("\nConfusion Matrix (% by row):\n", np.round(cm_percent, 1))
    
    # ---------- Macro ROC ----------
    fpr, tpr = dict(), dict()
    #  ROC
    for i in range(CONFIG.NUM_CLASSES): fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(all_targets_onehot[:, i], all_probs[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(CONFIG.NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(CONFIG.NUM_CLASSES): mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # 插值
    mean_tpr /= CONFIG.NUM_CLASSES
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    with open(PATH.ROC_CURVE_PATH, "w+") as f:
        f.write(str(fpr_macro.tolist()))
        f.write("\n")
        f.write(str(tpr_macro.tolist()))
    roc_auc_macro = sklearn.metrics.auc(fpr_macro, tpr_macro)
    plt.figure()
    plt.plot(fpr_macro, tpr_macro, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_macro:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # ---------- Macro PR ----------
    precision, recall = dict(), dict()
    for i in range(CONFIG.NUM_CLASSES): precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(all_targets_onehot[:, i], all_probs[:, i])
    # recall）
    all_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(all_recall)
    for i in range(CONFIG.NUM_CLASSES): mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])  
    mean_precision /= CONFIG.NUM_CLASSES
    precision_macro = mean_precision
    recall_macro = all_recall
    with open(PATH.PR_CURVE_PATH, "w+") as f:
        f.write(str(precision_macro.tolist()))
        f.write("\n")
        f.write(str(recall_macro.tolist()))
    pr_auc_macro = sklearn.metrics.auc(recall_macro, precision_macro)
    plt.figure()
    plt.plot(recall_macro, precision_macro, color='green', lw=2, label=f'PR Curve (AUC = {pr_auc_macro:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

    report = []
    report.append(f"Overall Acc: {oacc_score}")
    report.append(f"Balanced Acc: {bacc_score}")
    report.append(f"Macro F1-Score: {f1_macro}")
    report.append(f"Micro F1-Score: {f1_micro}")
    report.append(f"Weighted F1-Score: {f1_weighted}")
    report.append(f"ROC AUC (macro): {roc_auc_macro}")
    report.append(f"ROC AUC (micro): {roc_auc_micro}")
    report.append(f"PR AUC (macro): {pr_auc_macro}")
    report.append(f"PR AUC (micro): {pr_auc_micro}")
    report.append("\nConfusion Matrix (raw counts):\n" + str(conf_mat))
    report.append("\nConfusion Matrix (% by row):\n" + str(np.round(cm_percent, 1)))

    Confusion_PATH =PATH.Confusion_matrix_PATH   
    with open(Confusion_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return 

def train(train_loader, val_loader, test_loader,scalers):
    model = GetModel.get_model()
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU or CPU
    model = model.to(device) 
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.INITIAL_LR, weight_decay=1e-4) # 优化器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=16, cooldown=8, min_lr=1e-9)
    
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    metrics_func = torchmetrics.Accuracy(task="multiclass", num_classes=CONFIG.NUM_CLASSES).to(device)

    train_loss_avg_list, train_metrics_avg_list = [], []
    train_loss_std_list, train_metrics_std_list = [], []
    val_loss_avg_list, val_metrics_avg_list = [], []
    val_loss_std_list, val_metrics_std_list = [], []
    for epoch in range(CONFIG.EPOCHS):
        print('\nEpoch #{:03d}'.format(epoch + 1) + "/{:03d}".format(CONFIG.EPOCHS))
        
        train_loss_avg, train_metrics_avg, train_loss_std, train_metrics_std = train_func(train_loader, model, optimizer, loss_func=loss_func, metrics_func=metrics_func, device=device)
        val_loss_avg, val_metrics_avg, val_loss_std, val_metrics_std = val_func(val_loader, model, loss_func=loss_func, metrics_func=metrics_func, device=device)
        scheduler.step(val_loss_avg)

        print('Train_Loss={:.4f}, Train_Metrics={:.4f}, Val_Loss={:.4f}, Val_Metrics={:.4f}'.format(train_loss_avg, train_metrics_avg, val_loss_avg, val_metrics_avg))
        
        train_loss_avg_list.append(train_loss_avg)
        train_metrics_avg_list.append(train_metrics_avg)
        
        train_loss_std_list.append(train_loss_std)
        train_metrics_std_list.append(train_metrics_std)
        

        val_loss_avg_list.append(val_loss_avg)
        val_metrics_avg_list.append(val_metrics_avg)
        
        val_loss_std_list.append(val_loss_std)
        val_metrics_std_list.append(val_metrics_std)
        pass
    
    GetModel.save_model(model)
    MakeDataset.save_scalers(scalers) 
    DrawLog.write_log(train_loss_avg_list, train_metrics_avg_list, val_loss_avg_list, val_metrics_avg_list)
    
    test_func(test_loader, model, device=device)
    
    pass
    