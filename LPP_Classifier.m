%LPP Perform linearity preserving projection
%   [mappedX, mapping] = lpp(X, no_dims, k, sigma, eig_impl)
% X       待降维的数据集
% no_dims 降维后的维数，缺省值为2
% k       k个最近的邻居点，缺省值为12
% sigma   bandwidth of the Gaussian kernel (default = 1).
clear

load('orldata.mat');
facedatabase = double(orldata(1:50,:));     % 缩小简化样本
numClass = 40;        % 样本中有40个人
nsample_eachclass = 10;         % 每个人10张图
neachtrain = 5;     % 每个人取5张做训练样本
neachtest = 5;      % 每个人取5张做测试样本
% height = 112;       % 图的高
% width = 92;         % 图的宽
ori_dims = size(facedatabase,1);
%------------------训练数据集，ori_dims*(neachtest*nclass)的矩阵，trainingSet------------------
trainingSet = zeros(ori_dims, neachtrain*numClass);
for i = 1:numClass
    for j = 1:neachtrain
        trainingSet(:,(i-1)*neachtrain+j) = facedatabase(:,(i-1)*nsample_eachclass+2*j-1);
    end
end
%------------------测试数据集，ori_dims*(neachtest*nclass)的矩阵，trainingSet------------------
testingSet = zeros(ori_dims, neachtest*numClass);
for i = 1:numClass
    for j = 1:neachtrain
        testingSet(:,(i-1)*neachtest+j) = facedatabase(:,(i-1)*nsample_eachclass+2*j);
    end
end

numTrainInstance = size(trainingSet,2);    % 训练样本数
numTestInstance = size(testingSet,2);   % 测试样本数
perClassTrainLen = numTrainInstance/numClass;%每个类别的训练样本数
perClassTestLen = numTestInstance/numClass;%每个类别的测试样本数
T_trainingSet = trainingSet';
no_dims = 40;
% if size(T_training_set, 2) > size(T_training_set, 1)
%     error('Number of samples should be higher than number of dimensions.');
% end
if ~exist('no_dims', 'var')
    no_dims = 2;
end
if ~exist('k', 'var')
    k = 12;
end
if ~exist('sigma', 'var')
    sigma = 1;
end
if ~exist('eig_impl', 'var')
    eig_impl = 'Matlab';
end

% ------------------构造邻接图------------------
disp('Constructing neighborhood graph...');
if size(T_trainingSet, 1) < 4000
    G = L2_distance(T_trainingSet', T_trainingSet');  % 计算各样本间的距离
    % Compute neighbourhood graph
    [tmp, ind] = sort(G);       % 对每列元素升序排列后的矩阵放入tmp，ind记录原来的列位置
    for i=1:size(G, 1)
        G(i, ind((2 + k):end, i)) = 0;      % 2+knearest neighbors外的，置为零，即无边连接
    end
    G = sparse(double(G));      %将矩阵稀疏表示，因为有很多0
    G = max(G, G');             % 返回一个和A和B同大小的数组，其中的元素是从A或B中取出的最大元素。保证对称
else
    G = find_nn(T_trainingSet, k);
end
G = G .^ 2;
G = G ./ max(max(G));       % 返回每一列的最大元素，再求这些元素中最大的，即求矩阵中最大的元素

% Compute weights (W = G)
disp('Computing weight matrices...');

% Compute Gaussian kernel (heat kernel-based weights)
G(G ~= 0) = exp(-G(G ~= 0) / (2 * sigma ^ 2));                                                                                                                                  

% ------------------Construct diagonal weight matrix------------------
D = diag(sum(G, 2));        % 矩阵横向相加的和组成的列向量，组成对角阵

% Compute Laplacian
L = D - G;              % 拉普拉斯矩阵
L(isnan(L)) = 0; D(isnan(D)) = 0;       % 非数值的置为0
L(isinf(L)) = 0; D(isinf(D)) = 0;       % 无穷量的置为0

% Compute XDX and XLX and make sure these are symmetric
disp('Computing low-dimensional embedding...');
DP = T_trainingSet' * D * T_trainingSet;
LP = T_trainingSet' * L * T_trainingSet;
DP = (DP + DP') / 2;        % 确保对称
LP = (LP + LP') / 2;

% ------------------Perform eigenanalysis of generalized eigenproblem (as in LEM)------------------
if size(T_trainingSet, 1) >= 200 && no_dims < (size(T_trainingSet, 1) / 2) % 若样本数>200且降维后的特征<样本数的1/2
    if strcmp(eig_impl, 'JDQR')         % 比较两个字符串是否相等
        options.Disp = 0;
        options.LSolver = 'bicgstab';
        [eigvector, eigvalue] = jdqz(LP, DP, no_dims, 'SA', options);
    else
        options.disp = 0;
        options.issym = 1;
        options.isreal = 1;
        [eigvector, eigvalue] = eigs(LP, DP, no_dims, 'SA', options);
    end
else
    [eigvector, eigvalue] = eig(LP, DP);
end

% Sort eigenvalues in descending order and get smallest eigenvectors
[eigvalue, ind] = sort(diag(eigvalue), 'ascend');
projection = eigvector(:,ind(1:no_dims));

% Compute final linear basis and project data
allprojectionFace = (T_trainingSet * projection)';
%allprojectionFace.M = projection;
%allprojectionFace.mean = mean(training_set, 1);

%------------------利用1NN分类器进行分类，并统计正确率------------------
right = 0;
for x = 1:numTestInstance
    afterProjection = projection' * testingSet(:,x);
    error = zeros(numTrainInstance,1);
    for i=1:numTrainInstance
        %计算重构图像矩阵到各个类别图像矩阵间的距离
        dis = afterProjection -allprojectionFace(:,i);
        for j=1:size(dis,2)
            error(i) =error(i)+ norm(dis(:,j));
        end
    end;
    
    [errorS,errorIndex] = sort(error);  %对距离进行排序
    class = floor((errorIndex(1)-1)/perClassTrainLen)+1;%将图像分到距离最小的类别中去,预测的类别
    
    oriclass =  floor((x-1)/perClassTestLen)+1 ; %实际的类别
    if(class == oriclass)
        right = right+1;
    end
end

accuracy = right/numTestInstance;
disp('accuracy is');
accuracy
