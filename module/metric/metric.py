import torch
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score, recall_score, classification_report

def acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def uar(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        uar = recall_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), labels=[0, 1, 2, 3, 4, 5, 6], average='macro')
#     print(cf_report(output, target))
    return uar   

def roc_auc(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        y_true = target.detach().cpu().numpy()
        y_pred = pred.detach().cpu().numpy()
    try: 
        return roc_auc_score(y_true, y_pred)
    except:
        print('bad roc_auc score')
        return -1

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def cf_report(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return classification_report(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), labels=[0,1,2,3,4,5,6], target_names=['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Suprised'])