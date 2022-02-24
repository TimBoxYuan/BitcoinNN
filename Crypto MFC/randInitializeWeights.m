function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

%%%%%%%%%%*One effective strategy for choosing  
%%%%%%%%%%%%is to base it on the number of units in the network.
%%%%%%%%%%%A good choice of  is  where  and  are the number of units in 
%%%%%%%%%%%the layers adjacent to .
% root 6 / Root (L_out + L_in)


end

