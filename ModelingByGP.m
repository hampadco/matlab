function [Answer_Train,Answer_Test,Answer2,CTime] = ModelingByGP(Data)
for i = 1: size(Data,1)
   LM(i) = WekaM5TP_GP(Data(i,2), Data(i,3), Data(i,7), Data(i,1), Data(i,4), Data(i,5), Data(i,6));
end
%
nTrain = round(0.7*numel(LM));
LM_Train = LM(1:nTrain);
LM_Test = LM(nTrain+1:end);
Data = [Data(:,1:end-1),Data(:,1:end-1).^2,Data(:,1:end-1).^3,Data(:,end)];

Data_Train = Data(1:nTrain,:);
Data_Test = Data(nTrain+1:end,:);


Cluster_Size = unique(LM);
popusize = 100;
maxtreedepth = 4;
for c = 1:numel(Cluster_Size)
    index = find(LM_Train ==c);
    data = Data_Train(index,:);
    [s{c},tree{c}]=Gprogramming(data(:,1:end-1),data(:,end),popusize,maxtreedepth);
    txt = char(s{c});
    spaceIndx = strfind(char(txt),' ');
    txt = txt(spaceIndx(5)+1:end);
    txt = strrep(txt, newline, ' ');
    txt = regexprep(txt, '\s+', ' ');
    func_name = ['LM_',num2str(c)];
    vars = 'x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21';
    func_code = sprintf(['function y = %s(%s)\n' ...
                     '    y = %s;\n' ...
                     'end'], func_name, vars, txt);

     % Write to a .m file
     fid = fopen([func_name '.m'], 'w');
     fprintf(fid, '%s\n', func_code);
     fclose(fid);
end
CTime = toc;
for i = 1: size(Data,1)
   func = str2func(['LM_',num2str(LM(i))]);
   L_modeled_score(i,1) = func(Data(i,1),Data(i,2),Data(i,3),Data(i,4),Data(i,5),Data(i,6),Data(i,7),Data(i,8),Data(i,9),Data(i,10),Data(i,11),Data(i,12),Data(i,13),Data(i,14),Data(i,15),Data(i,16),Data(i,17),Data(i,18),Data(i,19),Data(i,20),Data(i,21));
end
L_modeled_score = L_modeled_score;
L_modeled = L_modeled_score;
% 
% nTrain = round(0.7*size(L_modeled_score,1));
%

L_modeled (L_modeled_score>=0.5) = 1;
L_modeled (L_modeled_score<0.5) = 0;
Target_Train = Data(1:nTrain,end);
Y_Train = L_modeled(1:nTrain,1);
Target_Test = Data(nTrain+1:end,end);
Y_test = L_modeled(nTrain+1:end,1);
Score_Train = L_modeled_score(1:nTrain,1);
Score_Test = L_modeled_score(nTrain+1:end,1);
%%
[confMat, order] = confusionmat(Target_Train, Y_Train);
TP_Train = confMat(2,2);
FP_Train = confMat(1,2);
TN_Train = confMat(1,1);
FN_Train = confMat(2,1);
Answer_Train = [TP_Train,FP_Train;FN_Train,TN_Train];
precision_train = TP_Train / (TP_Train + FP_Train);
OA_train = (TP_Train+TN_Train) / (TP_Train + TN_Train + FP_Train + FN_Train);
F1_Score_train = (2*TP_Train) / (2*TP_Train + FP_Train + FN_Train);
Recall_Train = TP_Train / (TP_Train + FN_Train);
MCC_Train = (TP_Train*TN_Train - FN_Train*FP_Train) / sqrt((TP_Train + FN_Train)*(TN_Train+FP_Train)*(TP_Train+FP_Train)*(TN_Train+FN_Train));
[confMat, order] = confusionmat(Target_Test, Y_test);
TP_Test = confMat(2,2);
FP_Test = confMat(1,2);
TN_Test = confMat(1,1);
FN_Test = confMat(2,1);
Answer_Test = [TP_Test,FP_Test;FN_Test,TN_Test];
precision_test = TP_Test / (TP_Test + FP_Test);
OA_test = (TP_Test+TN_Test) / (TP_Test + TN_Test + FP_Test + FN_Test);
F1_Score_test = (2*TP_Test) / (2*TP_Test + FP_Test + FN_Test);
Recall_Test = TP_Test / (TP_Test + FN_Test);
MCC_Test = (TP_Test*TN_Test - FN_Test*FP_Test) / sqrt((TP_Test + FN_Test)*(TN_Test+FP_Test)*(TP_Test+FP_Test)*(TN_Test+FN_Test));
%% Plot ROC
[X1,Y1,T,AUC_Train] = perfcurve(Target_Train, Score_Train, 1);  % '1' is the positive class
[X2,Y2,T,AUC_Test] = perfcurve(Target_Test, Score_Test, 1);  % '1' is the positive class
[X,Y,T,AUC_all] = perfcurve([Target_Train;Target_Test], [Score_Train;Score_Test], 1);  % '1' is the positive class
save('GP_results.mat')
figure;
plot(X, Y, 'b-', 'LineWidth', 2);
figure;
plot(X, Y, 'b-', 'LineWidth', 2)
hold on;
plot([0 1], [0 1], 'k--'); % reference line
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('GP ROC Curve (AUC = %.3f)', AUC_all));
grid on;
print(['ROC_AUC_GP'],'-dpng','-r300');  % saves as PNG
Answer2=[precision_train,OA_train,F1_Score_train,Recall_Train,MCC_Train,AUC_Train;precision_test,OA_test,F1_Score_test,Recall_Test,MCC_Test,AUC_Test];
close all

end