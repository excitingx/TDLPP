% TDLPP即two dimensional locality preservig projections

function [project_cell_dataSet, projection] = TDLPP(cell_dataSet, no_dims, k, sigma, eig_impl)
% cell_dataSet       待降维的数据集，cell类型
% no_dims 降维后的维数，缺省值为2
% k       k个最近的邻居点，缺省值为12
% sigma   bandwidth of the Gaussian kernel (default = 1).
% eig_impl = 'Matlab';
% Output: project_cell_dataSet 投影后的数据集，cell类型；projection 投影矩阵

num_dataSet = size(cell_dataSet,2);     % 样本数
height = size(cell_dataSet{1},1);        % 图像的高
%------------------构造邻接图G--------------------
disp('Constructing neighborhood graph...');
% 计算各样本间的距离
G = zeros(num_dataSet, num_dataSet);
for i = 1:num_dataSet
    for j = 1:num_dataSet
        diff = cell_dataSet{i} - cell_dataSet{j};
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
mat_dataSet = cell2mat(cell_dataSet');      % ve
I = eye(height);
DP = mat_dataSet' * kron(D, I) * mat_dataSet;
LP = mat_dataSet' * kron(L, I) * mat_dataSet;
DP = (DP + DP') / 2;        % 确保对称
LP = (LP + LP') / 2;
if size(cell_dataSet, 2) > 200 && no_dims < (size(T_trainingSet, 2) / 2) % 若样本数>200且降维后的特征<样本数的1/2
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
mat_projectionTrainingSet = (mat_dataSet * projection)';
% mat转cell：将投影后的矩阵分为1*numTrainInstance的cell，每个cell为no_dims*height大小
[Orin, Orim] = size(mat_projectionTrainingSet);
Block_n=no_dims*ones(1,Orin/no_dims);
Block_m=height*ones(1,Orim/height);
project_cell_dataSet = mat2cell(mat_projectionTrainingSet,Block_n,Block_m); 