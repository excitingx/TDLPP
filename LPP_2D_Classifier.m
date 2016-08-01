
%%%%%%%%%%%%%--------------2D LPP������ȡ������1NN���з���--------------%%%%%%%%%%%%%
clc
clear
%load face database
load('orldata.mat');
facedatabase = double(orldata);
numClass = 40;    % ��������40����
nsample_eachclass = 10;     % ÿ����10��ͼ
neachtrain = 5;     % ÿ����ȡ5����ѵ������
neachtest = 5;      % ÿ����ȡ5������������
height = 112;       % ͼ�ĸ�
width = 92;     % ͼ�Ŀ�
no_dims = 8;
%------------------ѵ�����ݼ���height*width*num�ľ���trainingSet------------------
%Ϊ������ܣ�Ԥ�����ڴ棬��ʼ������
trainingSet = zeros(height,width,neachtrain*numClass);
%��ԭʼ���ݼ��е�һά����ת��Ϊ��ά���󣬷�����ά�����У�����ά��ʾ�ڼ�������
for i = 1:numClass
    for j = 1:neachtrain
        %trainingVet(:,(i-1)*neachtrain+j) = facedatabase(:,(i-1)*nsample_eachclass+j*2-1);
        trainingSet(:,:,(i-1)*neachtrain+j) = reshape(facedatabase(:,(i-1)*nsample_eachclass+j),height,width);
    end
end
numTrainInstance = size(trainingSet,3);     % ѵ��������
perClassTrainLen = numTrainInstance/numClass;%ÿ������ѵ��������
for k = 1:numTrainInstance
    cell_trainingSet{k} = trainingSet(:,:,k);
end

%------------------�������ݣ�height*width*num�ľ���testingSet------------------
testingSet = zeros(height,width,neachtest*numClass);
for i = 1:numClass
    for j = 1:neachtest
        testingSet(:,:,(i-1)*neachtest+j) = reshape(facedatabase(:,(i-1)*nsample_eachclass+5+j),height,width);
    end
end
numTestInstance = size(testingSet,3);       % ���������� 
perClassTestLen = numTestInstance/numClass;%ÿ�����Ĳ���������
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
%------------------�����ڽ�ͼG------------------
disp('Constructing neighborhood graph...');
% �����������ľ���
G = zeros(numTrainInstance, numTrainInstance);
for i = 1:numTrainInstance
    for j = 1:numTrainInstance
        diff = cell_trainingSet{i} - cell_trainingSet{j};
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
mat_trainingSet = cell2mat(cell_trainingSet');      % ve
I = eye(height);
DP = mat_trainingSet' * kron(D, I) * mat_trainingSet;
LP = mat_trainingSet' * kron(L, I) * mat_trainingSet;
DP = (DP + DP') / 2;        % ȷ���Գ�
LP = (LP + LP') / 2;
if size(cell_trainingSet, 2) > 200 && no_dims < (size(T_trainingSet, 2) / 2) % ��������>200�ҽ�ά�������<��������1/2
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
mat_projectionTrainingSet = (mat_trainingSet * projection)';
% matתcell����ͶӰ��ľ����Ϊ1*numTrainInstance��cell��ÿ��cellΪno_dims*height��С
[Orin, Orim] = size(mat_projectionTrainingSet);
Block_n=no_dims*ones(1,Orin/no_dims);
Block_m=height*ones(1,Orim/height);
cell_projectionTrainingSet = mat2cell(mat_projectionTrainingSet,Block_n,Block_m);        

%------------------����1NN���������з��࣬��ͳ����ȷ��------------------
right = 0;
for x = 1:numTestInstance
    afterProjection = cell_testingSet{x}*projection;
    error = zeros(numTrainInstance,1);
    for i=1:numTrainInstance
        %�����ع�ͼ����󵽸������ͼ������ľ���
        miss = afterProjection -cell_projectionTrainingSet{i}';
        for j=1:size(miss,2)
            error(i) =error(i)+ norm(miss(:,j));
        end
    end;
    
    [errorS,errorIndex] = sort(error);  %�Ծ����������
    class = floor((errorIndex(1)-1)/perClassTrainLen)+1;%��ͼ��ֵ�������С�������ȥ,Ԥ������
    
    oriclass =  floor((x-1)/perClassTestLen)+1 ; %ʵ�ʵ����
    if(class == oriclass)
        right = right+1;
    end
end

accuracy = right/numTestInstance;
disp(['The accuracy is',num2str(accuracy)]);