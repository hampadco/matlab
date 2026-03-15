%% Run all 3 models: M5P, ANN, Random Forest
clear; clc; close all;

Data = table2array(readtable("data2.csv"));

%% Model 1: M5P (WEKA)
fprintf('Running M5P model...\n');
[Answer_Train_M5, Answer_Test_M5, Answer2_M5, Ctime_M5] = ModelingByM5(Data);
close all
fprintf('  M5P done. Time: %.2f s\n', Ctime_M5);

%% Model 2: ANN
fprintf('Running ANN model...\n');
[Answer_Train_ANN, Answer_Test_ANN, Answer2_ANN, Ctime_ANN] = ModelAnn_New(Data);
close all
fprintf('  ANN done. Time: %.2f s\n', Ctime_ANN);

%% Model 3: Random Forest
fprintf('Running Random Forest model...\n');
[Answer_Train_RF, Answer_Test_RF, Answer2_RF, Ctime_RF] = ModelRF(Data);
close all
fprintf('  RF done. Time: %.2f s\n', Ctime_RF);

%% Display Results
model_names = {'M5P', 'ANN', 'RF'};
metric_names = {'Precision', 'OA', 'F1-Score', 'Recall', 'MCC', 'AUC'};

fprintf('\n========== TRAIN RESULTS ==========\n');
fprintf('%-10s %10s %10s %10s %10s %10s %10s\n', 'Model', metric_names{:});
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'M5P', Answer2_M5(1,:));
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'ANN', Answer2_ANN(1,:));
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'RF', Answer2_RF(1,:));

fprintf('\n========== TEST RESULTS ===========\n');
fprintf('%-10s %10s %10s %10s %10s %10s %10s\n', 'Model', metric_names{:});
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'M5P', Answer2_M5(2,:));
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'ANN', Answer2_ANN(2,:));
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'RF', Answer2_RF(2,:));

fprintf('\n========== CONFUSION MATRICES (Test) ==========\n');
fprintf('M5P: TP=%d FP=%d FN=%d TN=%d\n', Answer_Test_M5(1,1), Answer_Test_M5(1,2), Answer_Test_M5(2,1), Answer_Test_M5(2,2));
fprintf('ANN: TP=%d FP=%d FN=%d TN=%d\n', Answer_Test_ANN(1,1), Answer_Test_ANN(1,2), Answer_Test_ANN(2,1), Answer_Test_ANN(2,2));
fprintf('RF:  TP=%d FP=%d FN=%d TN=%d\n', Answer_Test_RF(1,1), Answer_Test_RF(1,2), Answer_Test_RF(2,1), Answer_Test_RF(2,2));

fprintf('\n========== COMPUTATION TIME ==========\n');
fprintf('M5P: %.2f s | ANN: %.2f s | RF: %.2f s\n', Ctime_M5, Ctime_ANN, Ctime_RF);

%% Comparative ROC Plot
load('M5_results.mat', 'X', 'Y'); X_M5=X; Y_M5=Y; AUC_M5=Answer2_M5(2,6);
load('ANN_results.mat', 'X', 'Y'); X_ANN=X; Y_ANN=Y; AUC_ANN=Answer2_ANN(2,6);
load('RF_results.mat', 'X', 'Y'); X_RF=X; Y_RF=Y; AUC_RF=Answer2_RF(2,6);

figure('Position', [100 100 800 600]);
plot(X_M5, Y_M5, 'b-', 'LineWidth', 2); hold on;
plot(X_ANN, Y_ANN, 'r-', 'LineWidth', 2);
plot(X_RF, Y_RF, 'g-', 'LineWidth', 2);
plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Comparative ROC Curves');
legend(sprintf('M5P (AUC=%.3f)', AUC_M5), ...
       sprintf('ANN (AUC=%.3f)', AUC_ANN), ...
       sprintf('RF (AUC=%.3f)', AUC_RF), ...
       'Location', 'southeast');
grid on;
print('ROC_Comparative', '-dpng', '-r300');

%% Save all results
Total_Answer = [Answer_Train_M5, Answer_Test_M5; ...
                Answer_Train_ANN, Answer_Test_ANN; ...
                Answer_Train_RF, Answer_Test_RF];
All_Metrics = cat(3, Answer2_M5, Answer2_ANN, Answer2_RF);
save('All_Results_New.mat', 'Total_Answer', 'All_Metrics', ...
     'Answer2_M5', 'Answer2_ANN', 'Answer2_RF', ...
     'Answer_Train_M5', 'Answer_Test_M5', ...
     'Answer_Train_ANN', 'Answer_Test_ANN', ...
     'Answer_Train_RF', 'Answer_Test_RF', ...
     'Ctime_M5', 'Ctime_ANN', 'Ctime_RF');

fprintf('\nAll results saved to All_Results_New.mat\n');
