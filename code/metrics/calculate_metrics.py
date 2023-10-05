import sys
sys.path.append('../')
from metrics.binary_confusion_matrix import get_binary_confusion_matrix, get_threshold_binary_confusion_matrix 
from metrics.binary_statistical_metrics import get_accuracy, get_true_positive_rate, get_true_negative_rate, get_precision, get_f1_socre, get_iou
from metrics.dice_coefficient import hard_dice
from metrics.pr_curve import get_pr_curve
from metrics.roc_curve import get_auroc, get_roc_curve
from util.numpy_utils import tensor2numpy

def calculate_metrics(config,preds, targets,device):
    curr_TP, curr_FP, curr_TN, curr_FN = get_binary_confusion_matrix(
        input_=preds, target=targets, device =device, pixel = config.metrics.pixel, 
        threshold=config.metrics.threshold,
        reduction='sum')

    curr_acc = get_accuracy(true_positive=curr_TP,
                            false_positive=curr_FP,
                            true_negative=curr_TN,
                            false_negative=curr_FN)

    curr_recall = get_true_positive_rate(true_positive=curr_TP,
                                         false_negative=curr_FN)

    curr_specificity = get_true_negative_rate(false_positive=curr_FP,
                                              true_negative=curr_TN)

    curr_precision = get_precision(true_positive=curr_TP,
                                   false_positive=curr_FP)

    curr_f1_score = get_f1_socre(true_positive=curr_TP,
                                 false_positive=curr_FP,
                                 false_negative=curr_FN)

    curr_iou = get_iou(true_positive=curr_TP,
                       false_positive=curr_FP,
                       false_negative=curr_FN)

    curr_auroc = get_auroc(preds, targets)

    return (curr_acc, curr_recall, curr_specificity, curr_precision,
            curr_f1_score, curr_iou, curr_auroc)

def calculate_metrics_threshold(fusion_mat):
    mat = tensor2numpy(fusion_mat)
    true_positive_s, false_positive_s, _, false_negative_s = mat[:,:,0],mat[:,:,1],mat[:,:,2],mat[:,:,3]
    f1_per_image = (2 * true_positive_s) / (2 * true_positive_s +
                                    false_positive_s + false_negative_s)
    
    f1_max_per_image = f1_per_image.max(axis=1)
    OIS = f1_max_per_image.mean()       
    
    mat_1 = mat.sum(axis=0)
    true_positive, false_positive, true_negative, false_negative = mat_1[:,0],mat_1[:,1],mat_1[:,2],mat_1[:,3]
    f1_all_image = (2 * true_positive) / (2 * true_positive +
                                    false_positive + false_negative)
    prc = true_positive / (true_positive + false_positive)
    acc = (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative)
    iou = true_positive / (true_positive + false_positive + false_negative)
    rec= true_positive / (true_positive + false_negative)
    spe = true_negative / (true_negative + false_positive)
    dsc = 2 * true_positive / ( false_positive + 2 * true_positive + false_negative )
    AP = prc.mean()
    AIU = iou.mean()
    ODS = f1_all_image.max()
    accuracy = acc.mean()
    recall = rec.mean()
    specificity = spe.mean()
    DSC = dsc.mean()
    
    return (ODS, OIS, AIU, AP, accuracy,recall,specificity,DSC)