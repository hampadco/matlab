function [FuzzyResults] = FuzzyUncertainty(Data, N_samples)
%% Fuzzy Uncertainty Analysis based on alpha-cuts
%  Methodology: Ghasemi & Derakhshani (2021) + COV from Kumar et al. (2023)
%
%  Data:      Nx8 matrix [Z, N60, F, Si, Si1, Mw, a, L]
%  N_samples: number of Monte Carlo samples per alpha level (default: 100)
%
%  Returns FuzzyResults struct with min/max of each metric at each alpha

if nargin < 2
    N_samples = 100;
end

%% COV values from Kumar et al. (2023) Table 1
%  Columns: Z, N60, F, Si, Si1, Mw, a
COV = [0, 0.4, 0.35, 0.2, 0.2, 0.1, 0.2];

alpha_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0];
n_alpha = numel(alpha_levels);

metric_names = {'Precision', 'OA', 'F1', 'Recall', 'MCC', 'AUC'};
model_names = {'M5P', 'ANN', 'RF'};
n_metrics = 6;
n_models = 3;

%% Pre-train ANN and RF on original data for reuse
fprintf('Pre-training models on original data...\n');
Target = Data(:,end);
Input = Data(:,1:end-1);
NTrain = ceil(0.7*numel(Target));

Input_Train = Input(1:NTrain,:);
Input_Test = Input(NTrain+1:end,:);
Target_Train = Target(1:NTrain,1);
Target_Test = Target(NTrain+1:end,1);

%% Train ANN once
param.hiddenLayerSize = [10, 10];
param.hiddenLayerNum = 2;
param.TF = {'tansig','tansig','softmax'};
param.trainRatio = 0.70;
param.valRatio = 0.15;
param.testRatio = 0.15;
param.trainFcn = 'trainbr';
param.performParam.regularization = 0.2;
param.performFcn = 'crossentropy';
param.show = 1;
param.showWindow = 0;
param.showCommandLine = 0;
param.epochs = 2000;
param.goal = 0.0001;
param.max_fail = 200;

[~,~,~,~,net_trained] = ML_ANN(Input_Train, Input_Test, Target_Train, Target_Test, param);

%% Train RF once
rf_model = TreeBagger(100, Input_Train, Target_Train, ...
    'Method', 'classification', 'MinLeafSize', 5);

fprintf('Pre-training complete.\n');

%% Initialize result storage
% For each model and metric: store [min, max] at each alpha level
results_min = zeros(n_alpha, n_metrics, n_models);
results_max = zeros(n_alpha, n_metrics, n_models);

%% Main loop over alpha levels
for a_idx = 1:n_alpha
    alpha = alpha_levels(a_idx);
    fprintf('Processing alpha = %.1f (%d/%d)...\n', alpha, a_idx, n_alpha);

    delta_frac = (1 - alpha);

    if delta_frac == 0
        %% alpha=1: no uncertainty, just compute crisp values
        metrics_all = compute_all_metrics(Data, net_trained, rf_model, NTrain);
        for m = 1:n_models
            results_min(a_idx, :, m) = metrics_all(m, :);
            results_max(a_idx, :, m) = metrics_all(m, :);
        end
    else
        %% Monte Carlo sampling within alpha-cut bounds
        temp_min = inf(n_models, n_metrics);
        temp_max = -inf(n_models, n_metrics);

        for s = 1:N_samples
            Data_perturbed = Data;
            for col = 1:7
                if COV(col) > 0
                    delta = delta_frac * COV(col) * Data(:, col);
                    noise = -1 + 2*rand(size(Data,1), 1);
                    Data_perturbed(:, col) = Data(:, col) + delta .* noise;
                    Data_perturbed(:, col) = max(Data_perturbed(:, col), 0);
                end
            end

            metrics_all = compute_all_metrics(Data_perturbed, net_trained, rf_model, NTrain);

            for m = 1:n_models
                temp_min(m,:) = min(temp_min(m,:), metrics_all(m,:));
                temp_max(m,:) = max(temp_max(m,:), metrics_all(m,:));
            end
        end

        for m = 1:n_models
            results_min(a_idx, :, m) = temp_min(m, :);
            results_max(a_idx, :, m) = temp_max(m, :);
        end
    end
end

%% Package results
FuzzyResults.alpha_levels = alpha_levels;
FuzzyResults.metric_names = metric_names;
FuzzyResults.model_names = model_names;
FuzzyResults.results_min = results_min;
FuzzyResults.results_max = results_max;
FuzzyResults.COV = COV;
FuzzyResults.N_samples = N_samples;

%% Compute support widths (at alpha=0)
fprintf('\n========== FUZZY SUPPORT WIDTHS (alpha=0) ==========\n');
fprintf('%-10s', 'Model');
for k = 1:n_metrics
    fprintf('%12s', metric_names{k});
end
fprintf('\n');

for m = 1:n_models
    fprintf('%-10s', model_names{m});
    for k = 1:n_metrics
        width = results_max(end, k, m) - results_min(end, k, m);
        fprintf('%12.4f', width);
    end
    fprintf('\n');
end

save('FuzzyUncertainty_Results.mat', 'FuzzyResults');
fprintf('\nFuzzy uncertainty results saved to FuzzyUncertainty_Results.mat\n');

end

%% =========================================================================
function metrics_all = compute_all_metrics(Data, net_trained, rf_model, NTrain)
%% Computes metrics for all 3 models on given data.
%  Returns 3x6 matrix: [M5P; ANN; RF] x [Precision, OA, F1, Recall, MCC, AUC]

Target = Data(:, end);
Input = Data(:, 1:end-1);

Target_Train = Target(1:NTrain);
Target_Test = Target(NTrain+1:end);
Input_Train = Input(1:NTrain, :);
Input_Test = Input(NTrain+1:end, :);

metrics_all = zeros(3, 6);

%% M5P
try
    for i = 1:size(Data,1)
        Score_M5P(i,1) = WekaM5TP(Data(i,2), Data(i,3), Data(i,7), Data(i,1), Data(i,4), Data(i,5), Data(i,6));
    end
    Pred_M5P = double(Score_M5P >= 0.5);
    Score_Test_M5P = Score_M5P(NTrain+1:end);
    Pred_Test_M5P = Pred_M5P(NTrain+1:end);
    m = EvaluateModel(Target_Test, Pred_Test_M5P, Score_Test_M5P);
    metrics_all(1,:) = [m.Precision, m.OA, m.F1, m.Recall, m.MCC, m.AUC];
catch
    metrics_all(1,:) = NaN;
end

%% ANN
try
    Score_ANN_train = net_trained(Input_Train')';
    Score_ANN_test = net_trained(Input_Test')';
    Pred_ANN_test = double(Score_ANN_test >= 0.5);
    m = EvaluateModel(Target_Test, Pred_ANN_test, Score_ANN_test);
    metrics_all(2,:) = [m.Precision, m.OA, m.F1, m.Recall, m.MCC, m.AUC];
catch
    metrics_all(2,:) = NaN;
end

%% Random Forest
try
    [~, Score_RF_raw] = predict(rf_model, Input_Test);
    Score_RF_test = Score_RF_raw(:,2);
    Pred_RF_test = double(Score_RF_test >= 0.5);
    m = EvaluateModel(Target_Test, Pred_RF_test, Score_RF_test);
    metrics_all(3,:) = [m.Precision, m.OA, m.F1, m.Recall, m.MCC, m.AUC];
catch
    metrics_all(3,:) = NaN;
end

end
