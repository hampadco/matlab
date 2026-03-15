%% RunAll.m - Complete Pipeline: Models + Fuzzy Uncertainty + Fuzzy Reliability
%  Executes all phases in order and saves all results and figures
%  in the output folder with a descriptive summary.
clear; clc; close all;

fprintf('============================================================\n');
fprintf('  LIQUEFACTION PREDICTION: COMPLETE ANALYSIS PIPELINE\n');
fprintf('============================================================\n\n');

%% Create output folder
outDir = fullfile(pwd, 'Results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
fprintf('Output folder: %s\n\n', outDir);

%% Open log file
logFile = fullfile(outDir, 'analysis_log.txt');
diary(logFile);

%% Load Data
Data = table2array(readtable("data2.csv"));
[nSamples, nCols] = size(Data);
fprintf('Data loaded: %d samples, %d features + 1 target\n\n', nSamples, nCols-1);

%% ==================== PHASE 1: Run 3 Models ====================
fprintf('==================== PHASE 1: MODEL TRAINING ====================\n\n');

fprintf('[1/3] Running M5P (WEKA) model...\n');
[Answer_Train_M5, Answer_Test_M5, Answer2_M5, Ctime_M5] = ModelingByM5(Data);
close all;
movefile_safe('M5_results.mat', outDir);
movefile_safe('ROC_AUC_M5P.png', outDir);
fprintf('      M5P done. Time: %.2f s\n\n', Ctime_M5);

fprintf('[2/3] Running ANN model...\n');
[Answer_Train_ANN, Answer_Test_ANN, Answer2_ANN, Ctime_ANN] = ModelAnn_New(Data);
close all;
movefile_safe('ANN_results.mat', outDir);
movefile_safe('ROC_AUC_ANN.png', outDir);
fprintf('      ANN done. Time: %.2f s\n\n', Ctime_ANN);

fprintf('[3/3] Running Random Forest model...\n');
[Answer_Train_RF, Answer_Test_RF, Answer2_RF, Ctime_RF] = ModelRF(Data);
close all;
movefile_safe('RF_results.mat', outDir);
movefile_safe('ROC_AUC_RF.png', outDir);
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

fprintf('\n---------- CONFUSION MATRICES (Test) ----------\n');
fprintf('M5P: TP=%d FP=%d FN=%d TN=%d\n', Answer_Test_M5(1,1), Answer_Test_M5(1,2), Answer_Test_M5(2,1), Answer_Test_M5(2,2));
fprintf('ANN: TP=%d FP=%d FN=%d TN=%d\n', Answer_Test_ANN(1,1), Answer_Test_ANN(1,2), Answer_Test_ANN(2,1), Answer_Test_ANN(2,2));
fprintf('RF:  TP=%d FP=%d FN=%d TN=%d\n', Answer_Test_RF(1,1), Answer_Test_RF(1,2), Answer_Test_RF(2,1), Answer_Test_RF(2,2));

fprintf('\n---------- COMPUTATION TIME ----------\n');
fprintf('M5P: %.2f s | ANN: %.2f s | RF: %.2f s\n', Ctime_M5, Ctime_ANN, Ctime_RF);

%% Comparative ROC Plot
load(fullfile(outDir, 'M5_results.mat'), 'X', 'Y'); X_M5=X; Y_M5=Y;
load(fullfile(outDir, 'ANN_results.mat'), 'X', 'Y'); X_ANN=X; Y_ANN=Y;
load(fullfile(outDir, 'RF_results.mat'), 'X', 'Y'); X_RF=X; Y_RF=Y;

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
print(fullfile(outDir, 'ROC_Comparative'), '-dpng', '-r300');
close all;

%% Save Phase 1
save(fullfile(outDir, 'Phase1_ModelResults.mat'), ...
    'Answer2_M5', 'Answer2_ANN', 'Answer2_RF', ...
    'Answer_Train_M5', 'Answer_Test_M5', ...
    'Answer_Train_ANN', 'Answer_Test_ANN', ...
    'Answer_Train_RF', 'Answer_Test_RF', ...
    'Ctime_M5', 'Ctime_ANN', 'Ctime_RF');
fprintf('\nPhase 1 results saved.\n');

%% ==================== PHASE 2: Fuzzy Uncertainty ====================
fprintf('\n==================== PHASE 2: FUZZY UNCERTAINTY ====================\n\n');

N_samples = 100;
fprintf('Running fuzzy uncertainty analysis (N=%d samples per alpha)...\n', N_samples);
FuzzyResults = FuzzyUncertainty(Data, N_samples);
movefile_safe('FuzzyUncertainty_Results.mat', outDir);
close all;

fprintf('\nPlotting fuzzy triangles...\n');
PlotFuzzyTriangles(FuzzyResults);
movefile_safe('FuzzyTriangles_Metrics.png', outDir);
close all;
fprintf('Phase 2 complete.\n');

%% ==================== PHASE 3: Fuzzy Reliability ====================
fprintf('\n==================== PHASE 3: FUZZY RELIABILITY (beta & PL) ====================\n\n');

fprintf('Running fuzzy reliability analysis (N=%d samples per alpha)...\n', N_samples);
FuzzyRelResults = FuzzyReliability(Data, N_samples);
movefile_safe('FuzzyReliability_Results.mat', outDir);
movefile_safe('FuzzyReliability_Plots.png', outDir);
close all;
fprintf('Phase 3 complete.\n');

%% ==================== SAVE EVERYTHING ====================
save(fullfile(outDir, 'AllResults_Final.mat'));

%% ==================== WRITE DESCRIPTION FILE ====================
write_description_file(outDir, Answer2_M5, Answer2_ANN, Answer2_RF, ...
    Ctime_M5, Ctime_ANN, Ctime_RF, FuzzyResults, FuzzyRelResults, N_samples);

%% ==================== FINAL SUMMARY ====================
fprintf('\n============================================================\n');
fprintf('  ANALYSIS COMPLETE - ALL OUTPUTS IN: output/\n');
fprintf('============================================================\n\n');

diary off;
fprintf('Log saved to: %s\n', logFile);

%% =========================================================================
function movefile_safe(filename, destDir)
    if exist(filename, 'file')
        movefile(filename, fullfile(destDir, filename), 'f');
    end
end

%% =========================================================================
function write_description_file(outDir, A2_M5, A2_ANN, A2_RF, ...
    t_M5, t_ANN, t_RF, FR, FRR, N_samples)

    fid = fopen(fullfile(outDir, 'README_RESULTS.txt'), 'w', 'n', 'UTF-8');

    fprintf(fid, '============================================================\n');
    fprintf(fid, '  LIQUEFACTION PREDICTION - OUTPUT DESCRIPTION\n');
    fprintf(fid, '  Generated: %s\n', datestr(now));
    fprintf(fid, '============================================================\n\n');

    fprintf(fid, '=== PHASE 1: MODEL COMPARISON (M5P, ANN, Random Forest) ===\n\n');

    fprintf(fid, 'Files:\n');
    fprintf(fid, '  Phase1_ModelResults.mat  - All model metrics in MATLAB format\n');
    fprintf(fid, '  M5_results.mat           - M5P (WEKA) detailed results\n');
    fprintf(fid, '  ANN_results.mat          - ANN detailed results\n');
    fprintf(fid, '  RF_results.mat           - Random Forest detailed results\n');
    fprintf(fid, '  ROC_AUC_M5P.png          - ROC curve for M5P model\n');
    fprintf(fid, '  ROC_AUC_ANN.png          - ROC curve for ANN model\n');
    fprintf(fid, '  ROC_AUC_RF.png           - ROC curve for Random Forest model\n');
    fprintf(fid, '  ROC_Comparative.png      - All 3 ROC curves on one plot\n\n');

    fprintf(fid, 'Model Performance (Test Set):\n');
    fprintf(fid, '  %-8s %10s %10s %10s %10s %10s %10s %8s\n', ...
        'Model', 'Precision', 'OA', 'F1', 'Recall', 'MCC', 'AUC', 'Time(s)');
    fprintf(fid, '  %-8s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %8.2f\n', ...
        'M5P', A2_M5(2,:), t_M5);
    fprintf(fid, '  %-8s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %8.2f\n', ...
        'ANN', A2_ANN(2,:), t_ANN);
    fprintf(fid, '  %-8s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %8.2f\n\n', ...
        'RF', A2_RF(2,:), t_RF);

    fprintf(fid, '=== PHASE 2: FUZZY UNCERTAINTY ANALYSIS ===\n\n');

    fprintf(fid, 'Methodology: Ghasemi & Derakhshani (2021)\n');
    fprintf(fid, '  - Alpha-cut levels: 1.0, 0.8, 0.6, 0.4, 0.2, 0.0\n');
    fprintf(fid, '  - Monte Carlo samples per alpha: %d\n', N_samples);
    fprintf(fid, '  - COV values: Z=0, N60=0.4, F=0.35, Si=0.2, Si1=0.2, Mw=0.1, a=0.2\n\n');

    fprintf(fid, 'Files:\n');
    fprintf(fid, '  FuzzyUncertainty_Results.mat  - Fuzzy min/max for all metrics\n');
    fprintf(fid, '  FuzzyTriangles_Metrics.png    - 6 fuzzy triangle plots\n');
    fprintf(fid, '    (Precision, OA, F1, Recall, MCC, AUC)\n');
    fprintf(fid, '    Each plot shows 3 models: M5P (blue), ANN (red), RF (green)\n');
    fprintf(fid, '    Narrower triangle = more robust model\n\n');

    fprintf(fid, 'Fuzzy Support Widths (alpha=0, lower = better):\n');
    mn = FR.metric_names;
    fprintf(fid, '  %-8s', 'Model');
    for k = 1:numel(mn)
        fprintf(fid, '%12s', mn{k});
    end
    fprintf(fid, '\n');
    for m = 1:numel(FR.model_names)
        fprintf(fid, '  %-8s', FR.model_names{m});
        for k = 1:numel(mn)
            w = FR.results_max(end, k, m) - FR.results_min(end, k, m);
            fprintf(fid, '%12.4f', w);
        end
        fprintf(fid, '\n');
    end

    fprintf(fid, '\n=== PHASE 3: FUZZY RELIABILITY (Beta & PL) ===\n\n');

    fprintf(fid, 'Methodology: Kumar et al. (2023) formulas + Fuzzy approach\n');
    fprintf(fid, '  - CSR, CRR, FS computed using Idriss & Boulanger (2008)\n');
    fprintf(fid, '  - Beta = (mean(FS)-1) / std(FS)\n');
    fprintf(fid, '  - PL = 1 - normcdf(Beta)\n\n');

    fprintf(fid, 'Files:\n');
    fprintf(fid, '  FuzzyReliability_Results.mat  - Fuzzy beta, PL, FS results\n');
    fprintf(fid, '  FuzzyReliability_Plots.png    - Fuzzy triangles for Beta, PL, FS\n\n');

    fprintf(fid, 'Results:\n');
    fprintf(fid, '  Crisp Beta:  %.4f\n', FRR.beta_min(1));
    fprintf(fid, '  Crisp PL:    %.4f\n', FRR.PL_min(1));
    fprintf(fid, '  Fuzzy Beta range (alpha=0): [%.4f, %.4f]\n', FRR.beta_min(end), FRR.beta_max(end));
    fprintf(fid, '  Fuzzy PL range (alpha=0):   [%.4f, %.4f]\n\n', FRR.PL_min(end), FRR.PL_max(end));

    fprintf(fid, '=== OTHER FILES ===\n\n');
    fprintf(fid, '  AllResults_Final.mat   - Complete workspace (all variables)\n');
    fprintf(fid, '  analysis_log.txt       - Full console output log\n');
    fprintf(fid, '  README_RESULTS.txt     - This file\n\n');

    fprintf(fid, '============================================================\n');
    fprintf(fid, '  END OF DESCRIPTION\n');
    fprintf(fid, '============================================================\n');

    fclose(fid);
end
