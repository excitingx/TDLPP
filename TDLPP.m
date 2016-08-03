% TDLPP��two dimensional locality preservig projections

function [project_cell_dataSet, projection] = TDLPP(cell_dataSet, no_dims, k, sigma, eig_impl)
% cell_dataSet       ����ά�����ݼ���cell����
% no_dims ��ά���ά����ȱʡֵΪ2
% k       k��������ھӵ㣬ȱʡֵΪ12
% sigma   bandwidth of the Gaussian kernel (default = 1).
% eig_impl = 'Matlab';
% Output: project_cell_dataSet ͶӰ������ݼ���cell���ͣ�projection ͶӰ����

num_dataSet = size(cell_dataSet,2);     % ������
height = size(cell_dataSet{1},1);        % ͼ��ĸ�
%------------------�����ڽ�ͼG--------------------
disp('Constructing neighborhood graph...');
% �����������ľ���
G = zeros(num_dataSet, num_dataSet);
for i = 1:num_dataSet
    for j = 1:num_dataSet
        diff = cell_dataSet{i} - cell_dataSet{j};
        G(i,j) = sqrt( sum( diff(:).*diff(:) ) );
    end
end
[tmp, ind] = sort(G);
for i=1:size(G, 1)
    G(i, ind((2 + k):end, i)) = 0;      % 2+k nearest neighbors��ģ���Ϊ�㣬���ޱ�����
end
G = sparse(double(G));      %������ϡ���ʾ����Ϊ�кܶ�0
G = max(G, G');             % ����һ����A��Bͬ��С�����飬���е�Ԫ���Ǵ�A��B��ȡ�������Ԫ�ء���֤�Գ�
G = G .^ 2;
G = G ./ max(max(G));
%------------------ѡ��Ȩ��------------------
disp('Computing weight matrices...');
G(G ~= 0) = exp(-G(G ~= 0) / (2 * sigma ^ 2));
D = diag(sum(G, 2));
L = D - G;
L(isnan(L)) = 0; D(isnan(D)) = 0;
L(isinf(L)) = 0; D(isinf(D)) = 0;
%------------------��������ֵ����������,��ͶӰ����------------------
disp('Computing low-dimensional embedding...');
mat_dataSet = cell2mat(cell_dataSet');      % ve
I = eye(height);
DP = mat_dataSet' * kron(D, I) * mat_dataSet;
LP = mat_dataSet' * kron(L, I) * mat_dataSet;
DP = (DP + DP') / 2;        % ȷ���Գ�
LP = (LP + LP') / 2;
if size(cell_dataSet, 2) > 200 && no_dims < (size(T_trainingSet, 2) / 2) % ��������>200�ҽ�ά�������<��������1/2
    if strcmp(eig_impl, 'JDQR')         % �Ƚ������ַ����Ƿ����
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
% matתcell����ͶӰ��ľ����Ϊ1*numTrainInstance��cell��ÿ��cellΪno_dims*height��С
[Orin, Orim] = size(mat_projectionTrainingSet);
Block_n=no_dims*ones(1,Orin/no_dims);
Block_m=height*ones(1,Orim/height);
project_cell_dataSet = mat2cell(mat_projectionTrainingSet,Block_n,Block_m); 