function metrics = EvaluateModel(Target, Predicted, Score)
%% Computes all classification metrics from target, predicted labels, and scores.
%  Target:    Nx1 vector of true labels (0/1)
%  Predicted: Nx1 vector of predicted labels (0/1)
%  Score:     Nx1 vector of continuous scores (probability of class 1)
%
%  Returns struct with: Precision, OA, F1, Recall, MCC, AUC

[confMat, ~] = confusionmat(Target, Predicted);

if numel(confMat) < 4
    metrics.Precision = NaN;
    metrics.OA = NaN;
    metrics.F1 = NaN;
    metrics.Recall = NaN;
    metrics.MCC = NaN;
    metrics.AUC = NaN;
    return;
end

TP = confMat(2,2);
FP = confMat(1,2);
TN = confMat(1,1);
FN = confMat(2,1);

metrics.TP = TP;
metrics.FP = FP;
metrics.TN = TN;
metrics.FN = FN;

denom_prec = TP + FP;
metrics.Precision = TP / max(denom_prec, eps);

metrics.OA = (TP + TN) / (TP + TN + FP + FN);

metrics.F1 = (2*TP) / max(2*TP + FP + FN, eps);

metrics.Recall = TP / max(TP + FN, eps);

mcc_denom = sqrt((TP+FN)*(TN+FP)*(TP+FP)*(TN+FN));
metrics.MCC = (TP*TN - FN*FP) / max(mcc_denom, eps);

try
    [~,~,~,metrics.AUC] = perfcurve(Target, Score, 1);
catch
    metrics.AUC = NaN;
end

end
