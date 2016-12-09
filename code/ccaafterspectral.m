%% Initializations
numClusters=10;
dimension=8;
%Read spectral embedding file.
U = csvread('U1500.csv');
fu1 = csvread('features.csv');
%Choose Top 12 columns - low S.D.
fu = fu1(:,1:12);
%% CCA
[Wx, Wy, z] = cca(fu,U,dimension);
ccares = fu * Wx;
result1 = ccares;

%% Use seeds as Centroid Matrix
Centroid0 =  result1(5793,:) + result1(10805,:) + result1(2258,:)/3;
Centroid1 =  result1(8938,:) + result1(7423,:) + result1(1853,:)/3;
Centroid2 =  result1(4566,:) + result1(902,:) + result1(5633,:)/3;
Centroid3 =  result1(1529,:) + result1(9833,:) + result1(7709,:)/3;
Centroid4 =  result1(1942,:) + result1(935,:) + result1(4907,:)/3;
Centroid5 =  result1(6345,:) + result1(10950,:) + result1(2174,:)/3;
Centroid6 =  result1(9219,:) + result1(5958,:) + result1(9646,:)/3;
Centroid7 =  result1(2193,:) + result1(5757,:) + result1(6346,:)/3;
Centroid8 =  result1(5863,:) + result1(5838,:) + result1(9831,:)/3;
Centroid9 =  result1(9897,:) + result1(6314,:) + result1(2080,:)/3;
CentroidMatrix = [Centroid0;Centroid1;Centroid2;Centroid3;Centroid4;Centroid5;Centroid6;Centroid7;Centroid8;Centroid9];
X = transpose(result1);
%% Matlab K Means - Attempt 1

[idx_init,assignments_init] = kmeans(result1,10,'start',CentroidMatrix);


%% Try GMM - Attempt2
initMeans = transpose(assignments_init);
assignments = transpose(idx_init);
initSigmas = zeros(dimension,numClusters);
initWeights = zeros(1,numClusters);

for i=1:numClusters
  Xk = X(:,assignments==i);
  %initWeights(i) = size(Xk,dimension) / numClusters ;
  initWeights(i) = 0.1;
% Prepare Covariance Matrix from the data.
  if size(Xk,1) == 0 || size(Xk,2) == 0
    initSigmas(:,i) = diag(cov(X'));
  else
    initSigmas(:,i) = diag(cov(Xk'));
  end
end
[means,sigmas,weights,ll,posteriors] = vl_gmm(X, numClusters, ...
                                              'Initialization','custom', ...
                                              'InitMeans',initMeans, ...
                                              'InitCovariances',initSigmas, ...
                                              'InitPriors',initWeights, ...
                                              'Verbose', ...
                                              'MaxNumIterations', 10000,...
                                              'NumRepetitions',10) ;
%Pick the Cluster for which probability is highest.
[~,idx] = max(posteriors,[],1);
idx_init=idx;
%% Helper Code to check if the 30 seeds are all falling into same cluster
% and to check cluster sizes.
sol = idx_init;
sol_matrix = zeros(10,7);
sol_matrix(1,1)=5793;sol_matrix(1,2)=sol(5793);sol_matrix(1,3)=10805;sol_matrix(1,4)=sol(10805);sol_matrix(1,5)=2258;sol_matrix(1,6)=sol(2258);sol_matrix(1,7)=sum(idx_init==1);
sol_matrix(2,1)=8938;sol_matrix(2,2)=sol(8938);sol_matrix(2,3)=7423;sol_matrix(2,4)=sol(7423);sol_matrix(2,5)=1853;sol_matrix(2,6)=sol(1853);sol_matrix(2,7)=sum(idx_init==2);
sol_matrix(3,1)=4566;sol_matrix(3,2)=sol(4566);sol_matrix(3,3)=902;sol_matrix(3,4)=sol(902);sol_matrix(3,5)=5633;sol_matrix(3,6)=sol(5633);sol_matrix(3,7)=sum(idx_init==3);
sol_matrix(4,1)=1529;sol_matrix(4,2)=sol(1529);sol_matrix(4,3)=9833;sol_matrix(4,4)=sol(9833);sol_matrix(4,5)=7709;sol_matrix(4,6)=sol(7709);sol_matrix(4,7)=sum(idx_init==4);
sol_matrix(5,1)=1942;sol_matrix(5,2)=sol(1942);sol_matrix(5,3)=935;sol_matrix(5,4)=sol(935);sol_matrix(5,5)=4907;sol_matrix(5,6)=sol(4907);sol_matrix(5,7)=sum(idx_init==5);
sol_matrix(6,1)=6345;sol_matrix(6,2)=sol(6345);sol_matrix(6,3)=10950;sol_matrix(6,4)=sol(10950);sol_matrix(6,5)=2174;sol_matrix(6,6)=sol(2174);sol_matrix(6,7)=sum(idx_init==6);
sol_matrix(7,1)=9219;sol_matrix(7,2)=sol(9219);sol_matrix(7,3)=5958;sol_matrix(7,4)=sol(5958);sol_matrix(7,5)=9646;sol_matrix(7,6)=sol(9646);sol_matrix(7,7)=sum(idx_init==7);
sol_matrix(8,1)=2193;sol_matrix(8,2)=sol(2193);sol_matrix(8,3)=5757;sol_matrix(8,4)=sol(5757);sol_matrix(8,5)=6346;sol_matrix(8,6)=sol(6346);sol_matrix(8,7)=sum(idx_init==8);
sol_matrix(9,1)=5863;sol_matrix(9,2)=sol(5863);sol_matrix(9,3)=5838;sol_matrix(9,4)=sol(5838);sol_matrix(9,5)=9831;sol_matrix(9,6)=sol(9831);sol_matrix(9,7)=sum(idx_init==9);
sol_matrix(10,1)=9897;sol_matrix(10,2)=sol(9897);sol_matrix(10,3)=6314;sol_matrix(10,4)=sol(6314);sol_matrix(10,5)=2080;sol_matrix(10,6)=sol(2080);sol_matrix(10,7)=sum(idx_init==10);
%% Write CLustering Output to a file.
csvwrite('ClusterAssignments.csv',idx_init-1);
