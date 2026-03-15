function [FuzzyResults] = FuzzyUncertainty(Data, N_samples)
%% Fuzzy Uncertainty Analysis based on alpha-cuts
%  Methodology: Ghasemi & Derakhshani (2021) + COV from Kumar et al. (2023)
%
%  Data:      Nx8 matrix [Z, N60, F, Si, Si1, Mw, a, L]
%  N_samples: number of Monte Carlo samples per alpha level (default: 100)

if nargin < 2
    N_samples = 100;
end

%% COV values from Kumar et al. (2023) Table 1
COV = [0, 0.4, 0.35, 0.2, 0.2, 0.1, 0.2];

alpha_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0];
n_alpha = numel(alpha_levels);

metric_names = {'Precision', 'OA', 'F1', 'Recall', 'MCC', 'AUC'};
model_names = {'M5P', 'ANN', 'RF'};
n_metrics = 6;
n_models = 3;

%% Assign WEKA clusters on original data
fprintf('Assigning WEKA clusters...\n');
LM = zeros(1, size(Data,1));
for i = 1:size(Data,1)
    LM(i) = WekaM5TP_GP(Data(i,2), Data(i,3), Data(i,7), Data(i,1), Data(i,4), Data(i,5), Data(i,6));
end

Target = Data(:,end);
Input = Data(:,1:end-1);
NTrain = ceil(0.7*numel(Target));

Input_Train = Input(1:NTrain,:);
Input_Test = Input(NTrain+1:end,:);
Target_Train = Target(1:NTrain,1);
Target_Test = Target(NTrain+1:end,1);
LM_Train = LM(1:NTrain);
LM_Test = LM(NTrain+1:end);

%% Pre-train cluster-specific ANN models
fprintf('Pre-training ANN models per WEKA cluster...\n');
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

Cluster_IDs = unique(LM);
ann_nets = cell(1, max(Cluster_IDs));
rf_models = cell(1, max(Cluster_IDs));

for c = Cluster_IDs
    idx_train = find(LM_Train == c);
    idx_test = find(LM_Test == c);

    %% ANN per cluster
    if numel(idx_train) >= 5 && numel(unique(Target_Train(idx_train))) >= 2
        n_c = numel(idx_train);
        if n_c < 20
            param.hiddenLayerSize = [3, 2];
        elseif n_c < 50
            param.hiddenLayerSize = [5, 3];
        else
            param.hiddenLayerSize = [10, 5];
        end
        Inp_tr = Input_Train(idx_train,:);
        Tar_tr = Target_Train(idx_train);
        if isempty(idx_test)
            Inp_te = Inp_tr(1,:); Tar_te = Tar_tr(1);
        else
            Inp_te = Input_Test(idx_test,:);
            Tar_te = Target_Test(idx_test);
        end
        try
            [~,~,~,~,net] = ML_ANN(Inp_tr, Inp_te, Tar_tr, Tar_te, param);
            ann_nets{c} = net;
        catch
            ann_nets{c} = [];
        end
    end

    %% RF per cluster
    if numel(idx_train) >= 2 && numel(unique(Target_Train(idx_train))) >= 2
        rf_models{c} = TreeBagger(100, Input_Train(idx_train,:), Target_Train(idx_train), ...
            'Method', 'classification', 'MinLeafSize', 3);
    end
end
fprintf('Pre-training complete.\n');

%% Initialize result storage
results_min = zeros(n_alpha, n_metrics, n_models);
results_max = zeros(n_alpha, n_metrics, n_models);

%% Main loop over alpha levels
for a_idx = 1:n_alpha
    alpha = alpha_levels(a_idx);
    fprintf('Processing alpha = %.1f (%d/%d)...\n', alpha, a_idx, n_alpha);

    delta_frac = (1 - alpha);

    if delta_frac == 0
        metrics_all = compute_all_metrics(Data, LM, ann_nets, rf_models, NTrain);
        for m = 1:n_models
            results_min(a_idx, :, m) = metrics_all(m, :);
            results_max(a_idx, :, m) = metrics_all(m, :);
        end
    else
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

            metrics_all = compute_all_metrics(Data_perturbed, LM, ann_nets, rf_models, NTrain);

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
fprintf('\nFuzzy uncertainty results saved.\n');

end

%% =========================================================================
function metrics_all = compute_all_metrics(Data, LM, ann_nets, rf_models, NTrain)
%% Computes metrics for all 3 models on given (possibly perturbed) data.
%  Uses cluster assignments LM and pre-trained cluster-specific models.

Target = Data(:, end);
Input = Data(:, 1:end-1);

Target_Test = Target(NTrain+1:end);
Input_Test = Input(NTrain+1:end, :);
LM_Test = LM(NTrain+1:end);
nTest = numel(Target_Test);

metrics_all = zeros(3, 6);

%% M5P (uses its own branching logic, no pre-trained model needed)
try
    Score_M5P = zeros(size(Data,1), 1);
    for i = 1:size(Data,1)
        Score_M5P(i) = WekaM5TP(Data(i,2), Data(i,3), Data(i,7), Data(i,1), Data(i,4), Data(i,5), Data(i,6));
    end
    Pred_M5P = double(Score_M5P >= 0.5);
    m = EvaluateModel(Target_Test, Pred_M5P(NTrain+1:end), Score_M5P(NTrain+1:end));
    metrics_all(1,:) = [m.Precision, m.OA, m.F1, m.Recall, m.MCC, m.AUC];
catch
    metrics_all(1,:) = NaN;
end

%% ANN (cluster-specific)
try
    Score_ANN_test = zeros(nTest, 1);
    for i = 1:nTest
        c = LM_Test(i);
        if c <= numel(ann_nets) && ~isempty(ann_nets{c})
            out = ann_nets{c}(Input_Test(i,:)');
            Score_ANN_test(i) = out(1);
        else
            Score_ANN_test(i) = 0.5;
        end
    end
    Pred_ANN_test = double(Score_ANN_test >= 0.5);
    m = EvaluateModel(Target_Test, Pred_ANN_test, Score_ANN_test);
    metrics_all(2,:) = [m.Precision, m.OA, m.F1, m.Recall, m.MCC, m.AUC];
catch
    metrics_all(2,:) = NaN;
end

%% RF (cluster-specific)
try
    Score_RF_test = zeros(nTest, 1);
    for i = 1:nTest
        c = LM_Test(i);
        if c <= numel(rf_models) && ~isempty(rf_models{c})
            [~, sc] = predict(rf_models{c}, Input_Test(i,:));
            Score_RF_test(i) = sc(2);
        else
            Score_RF_test(i) = 0.5;
        end
    end
    Pred_RF_test = double(Score_RF_test >= 0.5);
    m = EvaluateModel(Target_Test, Pred_RF_test, Score_RF_test);
    metrics_all(3,:) = [m.Precision, m.OA, m.F1, m.Recall, m.MCC, m.AUC];
catch
    metrics_all(3,:) = NaN;
end

end
