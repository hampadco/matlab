function L = WekaM5TP_clustring(N60, F, a, Z, Si,Si1, Mw,idx)
%%

if idx == 1
if N60 <= 25.5
   if F <= 95.5 
      if a <= 0.19 
          LM=1;
       elseif a >  0.19
           LM=2;
      end
   elseif F >  95.5
      if Si <= 167.05
         if Mw <= 7.5 
             LM=3;
          elseif Mw >  7.5
              LM=4;
         end
      elseif Si >  167.05 
          LM=5;
      end
         end
elseif  N60 >  25.5
    LM=6;
 end

elseif idx == 2
  
if F <= 59.5  
   if N60 <= 29  
   if   a <= 0.19  
         if N60 <= 13.5 
            if Si1 <= 162.15 
               LM=1;
            elseif Si1 >  162.15 
                LM=2;
            end
         elseif N60 >  13.5 
             LM=3;
            end
   elseif   a >  0.19 
         if N60 <= 21.5 
             LM=4;
          elseif N60 >  21.5 
              LM=5;
            end
    end
   elseif N60 >  29 
       LM=6;
    end
elseif F >  59.5
    LM=7;
end
  else
if F <= 93.5  
if    Mw <= 6.915 
    LM=1;
elseif   Mw >  6.915 
      if Z <= 2.05 
          LM=2;
          elseif Z >  2.05
         if N60 <= 12.85 
             LM=3;
             elseif N60 >  12.85 
            if N60 <= 22.5
                LM=4;
                elseif N60 >  22.5 
                    LM=5;
                end
             end
          end
       end
elseif F >  93.5 
    LM=6;
end
end
%%
if idx == 1
    if LM==1
       L = -0.0017 * Z - 0.0531 * N60 - 0.0049 * F + 0.003 * Si - 0.0017 * Si1 - 0.0462 * Mw + 0.3848 * a + 1.288;
    elseif LM==2
       L = 0.0588 * Z - 0.0151 * N60 - 0.0022 * F - 0.0022 * Si - 0.0044 * Si1 - 0.0171 * Mw + 0.233 * a + 1.4155;
    elseif LM==3
       L = -0.0535 * Z + 0.0154 * N60 - 0.0013 * F + 0.0033 * Si1 - 0.6321 * Mw + 0.2524 * a + 5.1689;
    elseif LM==4
        L = -0.0772 * Z + 0.0001 * N60 - 0.0013 * F + 0.0056 * Si1 - 0.9324 * Mw + 0.2524 * a + 7.4656;
    elseif LM==5
        L = -0.0355 * Z + 0.0025 * N60 - 0.0013 * F + 0.0009 * Si1 - 0.3922 * Mw + 0.2524 * a + 3.3527;
    elseif LM==6
        L = -0.005 * Z - 0.0047 * N60 - 0.0008 * F + 0.1804 * a + 0.1892;
    end
elseif idx==2
  if LM==1 
    L = -0.0257 * Z - 0.006 * N60 - 0.0024 * F - 0.0026 * Si1 + 0.5528 * a + 1.232;

  elseif LM==2
    L = -0.0257 * Z - 0.0093 * N60 - 0.0024 * F - 0.0028 * Si1 + 0.5528 * a + 1.2666;
  elseif LM==3
    L = -0.0257 * Z - 0.0204 * N60 - 0.0024 * F - 0.0006 * Si1 + 0.5528 * a + 0.9092;
  elseif LM==4
    L = -0.0414 * Z - 0.0258 * N60 - 0.0061 * F + 0.5976 * a + 1.5993;
  elseif LM==5
    L = -0.0575 * Z - 0.0252 * N60 - 0.0098 * F + 0.5976 * a + 1.7972;
  elseif LM==6
    L = -0.0186 * Z - 0.0087 * N60 - 0.0007 * F + 0.3699 * a + 0.4884;
  elseif LM==7
    L = -0.0078 * Z - 0.0036 * N60 - 0.0011 * F + 0.1528 * a + 0.244;
  end
elseif idx==3
  if LM==1
    L = 0.0212 * Z - 0.0229 * N60 - 0.0008 * F + 0.0015 * Si - 0.003 * Si1 + 0.0472 * Mw + 1.3753 * a - 0.01;
  elseif LM==2
    L = 0.0212 * Z - 0.0173 * N60 - 0.0013 * F + 0.0477 * Si - 0.0484 * Si1 - 0.2901 * Mw + 0.164 * a + 2.5799;
  elseif LM==3
    L = 0.102 * Z + 0.0219 * N60 - 0.0031 * F + 0.0023 * Si - 0.0155 * Si1 + 0.0225 * Mw + 0.1576 * a + 0.7147;
  elseif LM==4
    L = 0.0662 * Z - 0.0133 * N60 - 0.002 * F + 0.0052 * Si - 0.0076 * Si1 + 0.0225 * Mw + 0.694 * a + 0.2877;
  elseif LM==5
    L = 0.0142 * Z - 0.0197 * N60 - 0.0026 * F + 0.0076 * Si - 0.0105 * Si1 + 0.0225 * Mw + 0.6087 * a + 0.387;
  elseif LM==6
    L = 0.0653 * Z - 0.0028 * N60 - 0.0008 * F + 0.0005 * Si - 0.0016 * Si1 + 0.0296 * Mw + 0.0913 * a - 0.1126;
  end
end