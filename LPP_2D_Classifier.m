
%%%%%%%%%%%%%--------------2D LPP特征提取，并用1NN进行分类--------------%%%%%%%%%%%%%
clc
clear
%load face database
load('orldata.mat');
facedatabase = double(orldata);
numClass = 40;    % 样本中有40个人
nsample_eachclass = 10;     % 每个人10张图
neachtrain = 5;     % 每个人取5张做训练样本
neachtest = 5;      % 每个人取5张做测试样本
height = 112;       % 图的高
width = 92;     % 图的宽
no_dims = 8;
%------------------训练数据集，height*width*num的矩阵，trainingSet------------------
%为提高性能，预分配内存，初始化矩阵
trainingSet = zeros(height,width,neachtrain*numClass);
%将原始数据集中的一维向量转换为二维矩阵，放入三维矩阵中，第三维表示第几个样本
for i = 1:numClass
    for j = 1:neachtrain
        %trainingVet(:,(i-1)*neachtrain+j) = facedatabase(:,(i-1)*nsample_eachclass+j*2-1);
        trainingSet(:,:,(i-1)*neachtrain+j) = reshape(facedatabase(:,(i-1)*nsample_eachclass+j),height,width);
    end
end
numTrainInstance = size(trainingSet,3);     % 训练样本数
perClassTrainLen = numTrainInstance/numClass;%每个类别的训练样本数
for k = 1:numTrainInstance
    cell_trainingSet{k} = trainingSet(:,:,k);
end

%------------------测试数据，height*width*num的矩阵，testingSet------------------
testingSet = zeros(height,width,neachtest*numClass);
for i = 1:numClass
    for j = 1:neachtest
        testingSet(:,:,(i-1)*neachtest+j) = reshape(facedatabase(:,(i-1)*nsample_eachclass+5+j),height,width);
    end
end
numTestInstance = size(testingSet,3);       % 测试样本数 
perClassTestLen = numTestInstance/numClass;%每个类别的测试样本数
for k = 1:numTestInstance
    cell_testingSet{k} = testingSet(:,:,k);
end


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
%-----------------------------------------------------------2D LPP-----------------------------------------------------------
%------------------构造邻接图G------------------
disp('Constructing neighborhood graph...');
% 计算各样本间的距离
G = zeros(numTrainInstance, numTrainInstance);
for i = 1:numTrainInstance
    for j = 1:numTrainInstance
        diff = cell_trainingSet{i} - cell_trainingSet{j};
        G(i,j) = sqrt( sum( diff(:).*diff(:) ) );
    end
end
[tmp, ind] = sort(G);
for i=1:size(G, 1)
    G(i, ind((2 + k):end, i)) = 0;      % 2+k nearest neighbors外的，置为零，即无边连接
end
G = sparse(double(G));      %将矩阵稀疏表示，因为有很多0
G = max(G, G');             % 返回一个和A和B同大小的数组，其中的元素是从A或B中取出的最大元素。保证对称
G = G .^ 2;
G = G ./ max(max(G));
%------------------选择权重------------------
disp('Computing weight matrices...');
G(G ~= 0) = exp(-G(G ~= 0) / (2 * sigma ^ 2));
D = diag(sum(G, 2));
L = D - G;
L(isnan(L)) = 0; D(isnan(D)) = 0;
L(isinf(L)) = 0; D(isinf(D)) = 0;
%------------------计算特征值、特征向量,求投影矩阵------------------
disp('Computing low-dimensional embedding...');
mat_trainingSet = cell2mat(cell_trainingSet');      % ve
I = eye(height);
DP = mat_trainingSet' * kron(D, I) * mat_trainingSet;
LP = mat_trainingSet' * kron(L, I) * mat_trainingSet;
DP = (DP + DP') / 2;        % 确保对称
LP = (LP + LP') / 2;
if size(cell_trainingSet, 2) > 200 && no_dims < (size(T_trainingSet, 2) / 2) % 若样本数>200且降维后的特征<样本数的1/2
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
[eigvalue, ind] = sort(diag(eigvalue), 'ascend');
projection = eigvector(:,ind(1:no_dims));
mat_projectionTrainingSet = (mat_trainingSet * projection)';
% mat转cell：将投影后的矩阵分为1*numTrainInstance的cell，每个cell为no_dims*height大小
[Orin, Orim] = size(mat_projectionTrainingSet);
Block_n=no_dims*ones(1,Orin/no_dims);
Block_m=height*ones(1,Orim/height);
cell_projectionTrainingSet = mat2cell(mat_projectionTrainingSet,Block_n,Block_m);        

%------------------利用1NN分类器进行分类，并统计正确率------------------
right = 0;
for x = 1:numTestInstance
    afterProjection = cell_testingSet{x}*projection;
    error = zeros(numTrainInstance,1);
    for i=1:numTrainInstance
        %计算重构图像矩阵到各个类别图像矩阵间的距离
        miss = afterProjection -cell_projectionTrainingSet{i}';
        for j=1:size(miss,2)
            error(i) =error(i)+ norm(miss(:,j));
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
disp(['The accuracy is',num2str(accuracy)]);