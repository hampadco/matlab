function [Y_Train,Y_test,Y_Train_Target,Y_test_Target,net]=ML_ANN(Input_Train,Input_Test, Output_Train,Output_Test,param)
%%
inputs = [Input_Train;Input_Test]';
targets = [Output_Train;Output_Test]';
% Create a Fitting Network
hiddenLayerSize = param.hiddenLayerSize;
hiddenLayerNum = param.hiddenLayerNum;
TF=param.TF;
net = patternnet(hiddenLayerSize);
% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'divideblock';  % Divide data as block
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = param.trainRatio;
net.divideParam.valRatio = param.valRatio;
net.divideParam.testRatio = param.testRatio;
net.performParam.regularization = param.performParam.regularization;
% For help on training function 'trainlm' type: help trainlm
% For a list of all training functions type: help nntrain
net.trainFcn = param.trainFcn;  % Levenberg-Marquardt

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = param.performFcn;  % Mean squared error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','ploterrhist','plotregression','plotfit'};

net.trainParam.showWindow=param.showWindow;
net.trainParam.showCommandLine=param.showCommandLine;
param.show
net.trainParam.show=param.show;
net.trainParam.epochs=param.epochs;
net.trainParam.goal=param.goal;
net.trainParam.max_fail=param.max_fail;

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

% Recalculate Training, Validation and Test Performance
trainInd=tr.trainInd;
trainInputs = inputs(:,trainInd);
trainTargets = targets(:,trainInd);
trainOutputs = outputs(:,trainInd);
trainErrors = trainTargets-trainOutputs;
trainPerformance = perform(net,trainTargets,trainOutputs);

valInd=tr.valInd;
valInputs = inputs(:,valInd);
valTargets = targets(:,valInd);
valOutputs = outputs(:,valInd);
valErrors = valTargets-valOutputs;
valPerformance = perform(net,valTargets,valOutputs);

testInd=tr.testInd;
testInputs = inputs(:,testInd);
testTargets = targets(:,testInd);
testOutputs = outputs(:,testInd);
testError = testTargets-testOutputs;
testPerformance = perform(net,testTargets,testOutputs);

Y_Train=trainOutputs';
Y_test=[valOutputs';testOutputs'];
Y_Train_Target=trainTargets';
Y_test_Target=[valTargets';testTargets'];
end