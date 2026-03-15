%% RunAll.m - Complete Pipeline: Models + Fuzzy Uncertainty + Fuzzy Reliability
%  Executes all phases in order and saves all results and figures.
clear; clc; close all;
fprintf('============================================================\n');
fprintf('  LIQUEFACTION PREDICTION: COMPLETE ANALYSIS PIPELINE\n');
fprintf('============================================================\n\n');

%% Load Data
Data = table2array(readtable("data2.csv"));
fprintf('Data loaded: %d samples, %d features + 1 target\n\n', size(Data,1), size(Data,2)-1);

%% ==================== PHASE 1: Run 3 Models ====================
fprintf('==================== PHASE 1: MODEL TRAINING ====================\n\n');

fprintf('[1/3] Running M5P (WEKA) model...\n');
[Answer_Train_M5, Answer_Test_M5, Answer2_M5, Ctime_M5] = ModelingByM5(Data);
close all;
fprintf('      M5P done. Time: %.2f s\n\n', Ctime_M5);

fprintf('[2/3] Running ANN model...\n');
[Answer_Train_ANN, Answer_Test_ANN, Answer2_ANN, Ctime_ANN] = ModelAnn_New(Data);
close all;
fprintf('      ANN done. Time: %.2f s\n\n', Ctime_ANN);

fprintf('[3/3] Running Random Forest model...\n');
[Answer_Train_RF, Answer_Test_RF, Answer2_RF, Ctime_RF] = ModelRF(Data);
close all;
fprintf('      RF done. Time: %.2f s\n\n', Ctime_RF);

%% Display Phase 1 Results
metric_names = {'Precision', 'OA', 'F1-Score', 'Recall', 'MCC', 'AUC'};

fprintf('---------- TRAIN RESULTS ----------\n');
fprintf('%-10s %10s %10s %10s %10s %10s %10s\n', 'Model', metric_names{:});
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'M5P', Answer2_M5(1,:));
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'ANN', Answer2_ANN(1,:));
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'RF', Answer2_RF(1,:));

fprintf('\n---------- TEST RESULTS -----------\n');
fprintf('%-10s %10s %10s %10s %10s %10s %10s\n', 'Model', metric_names{:});
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'M5P', Answer2_M5(2,:));
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'ANN', Answer2_ANN(2,:));
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n', 'RF', Answer2_RF(2,:));

%% Comparative ROC
load('M5_results.mat', 'X', 'Y'); X_M5=X; Y_M5=Y;
load('ANN_results.mat', 'X', 'Y'); X_ANN=X; Y_ANN=Y;
load('RF_results.mat', 'X', 'Y'); X_RF=X; Y_RF=Y;

figure('Position', [100 100 800 600]);
plot(X_M5, Y_M5, 'b-', 'LineWidth', 2); hold on;
plot(X_ANN, Y_ANN, 'r-', 'LineWidth', 2);
plot(X_RF, Y_RF, 'g-', 'LineWidth', 2);
plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Comparative ROC Curves');
legend(sprintf('M5P (AUC=%.3f)', Answer2_M5(2,6)), ...
       sprintf('ANN (AUC=%.3f)', Answer2_ANN(2,6)), ...
       sprintf('RF (AUC=%.3f)', Answer2_RF(2,6)), ...
       'Location', 'southeast');
grid on;
print('ROC_Comparative', '-dpng', '-r300');
close all;

%% Save Phase 1
save('Phase1_ModelResults.mat', ...
    'Answer2_M5', 'Answer2_ANN', 'Answer2_RF', ...
    'Answer_Train_M5', 'Answer_Test_M5', ...
    'Answer_Train_ANN', 'Answer_Test_ANN', ...
    'Answer_Train_RF', 'Answer_Test_RF', ...
    'Ctime_M5', 'Ctime_ANN', 'Ctime_RF');
fprintf('\nPhase 1 results saved to Phase1_ModelResults.mat\n');

%% ==================== PHASE 2: Fuzzy Uncertainty ====================
fprintf('\n==================== PHASE 2: FUZZY UNCERTAINTY ====================\n\n');

N_samples = 100;
fprintf('Running fuzzy uncertainty analysis (N=%d samples per alpha)...\n', N_samples);
FuzzyResults = FuzzyUncertainty(Data, N_samples);
close all;

fprintf('\nPlotting fuzzy triangles...\n');
PlotFuzzyTriangles(FuzzyResults);
close all;
fprintf('Phase 2 complete.\n');

%% ==================== PHASE 3: Fuzzy Reliability ====================
fprintf('\n==================== PHASE 3: FUZZY RELIABILITY (beta & PL) ====================\n\n');

fprintf('Running fuzzy reliability analysis (N=%d samples per alpha)...\n', N_samples);
FuzzyRelResults = FuzzyReliability(Data, N_samples);
close all;
fprintf('Phase 3 complete.\n');

%% ==================== FINAL SUMMARY ====================
fprintf('\n============================================================\n');
fprintf('  ANALYSIS COMPLETE\n');
fprintf('============================================================\n\n');
fprintf('Generated files:\n');
fprintf('  - Phase1_ModelResults.mat      (model metrics)\n');
fprintf('  - FuzzyUncertainty_Results.mat (fuzzy uncertainty)\n');
fprintf('  - FuzzyReliability_Results.mat (fuzzy beta/PL)\n');
fprintf('  - ROC_Comparative.png          (comparative ROC)\n');
fprintf('  - ROC_AUC_M5P.png              (M5P ROC)\n');
fprintf('  - ROC_AUC_ANN.png              (ANN ROC)\n');
fprintf('  - ROC_AUC_RF.png               (RF ROC)\n');
fprintf('  - FuzzyTriangles_Metrics.png   (fuzzy triangles)\n');
fprintf('  - FuzzyReliability_Plots.png   (beta/PL triangles)\n');
fprintf('\n');

save('AllResults_Final.mat');
fprintf('All workspace saved to AllResults_Final.mat\n');
