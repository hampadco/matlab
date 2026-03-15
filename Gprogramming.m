function [s,tree]=Gprogramming(X,Y,popusize,maxtreedepth)
addpath('GpOls\')

%GP equation symbols
symbols{1} = {'.*','./','-','+'};
symbols{2} = {'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21'};

%Initial population
popu = gpols_init(popusize,maxtreedepth,symbols);

%first evaluation
opt = [0.8 0.6 0.5 5 2 10 10 0.5 1 1];
popu = gpols_evaluate(popu,[1:popusize],X,Y,[],opt(6:9));
%info
disp(gpols_result([],0));
disp(gpols_result(popu,1));
%GP loops
for c = 2:100,
  %iterate 
  popu = gpols_mainloop(popu,X,Y,[],opt);
  %info  
  disp(gpols_result(popu,1));
end

%Result
[s,tree] = gpols_result(popu,2);
disp(s);
end
