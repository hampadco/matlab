%%
Data = table2array(readtable("data.csv"));
[I_train, I_val, I_test] = dividerand(828,0.7,0.15,0.15);
I=[I_train';I_val';I_test'];
Data_Train = Data(I_train,:);
Data_Test = Data([I_val';I_test';],:);
Mean_Train = mean(Data_Train);
Mean_Test = mean(Data_Test);
save("I.mat","I");
Data = Data(I,:);
save("Data.mat","Data");