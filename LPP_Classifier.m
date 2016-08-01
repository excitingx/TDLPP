%LPP Perform linearity preserving projection
%   [mappedX, mapping] = lpp(X, no_dims, k, sigma, eig_impl)
% X       ����ά�����ݼ�
% no_dims ��ά���ά����ȱʡֵΪ2
% k       k��������ھӵ㣬ȱʡֵΪ12
% sigma   bandwidth of the Gaussian kernel (default = 1).
clear

load('orldata.mat');
facedatabase = double(orldata(1:50,:));     % ��С������
numClass = 40;        % ��������40����
nsample_eachclass = 10;         % ÿ����10��ͼ
neachtrain = 5;     % ÿ����ȡ5����ѵ������
neachtest = 5;      % ÿ����ȡ5������������
% height = 112;       % ͼ�ĸ�
% width = 92;         % ͼ�Ŀ�
ori_dims = size(facedatabase,1);
%------------------ѵ�����ݼ���ori_dims*(neachtest*nclass)�ľ���trainingSet------------------
trainingSet = zeros(ori_dims, neachtrain*numClass);
for i = 1:numClass
    for j = 1:neachtrain
        trainingSet(:,(i-1)*neachtrain+j) = facedatabase(:,(i-1)*nsample_eachclass+2*j-1);
    end
end
%------------------�������ݼ���ori_dims*(neachtest*nclass)�ľ���trainingSet------------------
testingSet = zeros(ori_dims, neachtest*numClass);
for i = 1:numClass
    for j = 1:neachtrain
        testingSet(:,(i-1)*neachtest+j) = facedatabase(:,(i-1)*nsample_eachclass+2*j);
    end
end

numTrainInstance = size(trainingSet,2);    % ѵ��������
numTestInstance = size(testingSet,2);   % ����������
perClassTrainLen = numTrainInstance/numClass;%ÿ������ѵ��������
perClassTestLen = numTestInstance/numClass;%ÿ�����Ĳ���������
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

% ------------------�����ڽ�ͼ------------------
disp('Constructing neighborhood graph...');
if size(T_trainingSet, 1) < 4000
    G = L2_distance(T_trainingSet', T_trainingSet');  % �����������ľ���
    % Compute neighbourhood graph
    [tmp, ind] = sort(G);       % ��ÿ��Ԫ���������к�ľ������tmp��ind��¼ԭ������λ��
    for i=1:size(G, 1)
        G(i, ind((2 + k):end, i)) = 0;      % 2+knearest neighbors��ģ���Ϊ�㣬���ޱ�����
    end
    G = sparse(double(G));      %������ϡ���ʾ����Ϊ�кܶ�0
    G = max(G, G');             % ����һ����A��Bͬ��С�����飬���е�Ԫ���Ǵ�A��B��ȡ�������Ԫ�ء���֤�Գ�
else
    G = find_nn(T_trainingSet, k);
end
G = G .^ 2;
G = G ./ max(max(G));       % ����ÿһ�е����Ԫ�أ�������ЩԪ�������ģ��������������Ԫ��

% Compute weights (W = G)
disp('Computing weight matrices...');

% Compute Gaussian kernel (heat kernel-based weights)
G(G ~= 0) = exp(-G(G ~= 0) / (2 * sigma ^ 2));                                                                                                                                  

% ------------------Construct diagonal weight matrix------------------
D = diag(sum(G, 2));        % ���������ӵĺ���ɵ�����������ɶԽ���

% Compute Laplacian
L = D - G;              % ������˹����
L(isnan(L)) = 0; D(isnan(D)) = 0;       % ����ֵ����Ϊ0
L(isinf(L)) = 0; D(isinf(D)) = 0;       % ����������Ϊ0

% Compute XDX and XLX and make sure these are symmetric
disp('Computing low-dimensional embedding...');
DP = T_trainingSet' * D * T_trainingSet;
LP = T_trainingSet' * L * T_trainingSet;
DP = (DP + DP') / 2;        % ȷ���Գ�
LP = (LP + LP') / 2;

% ------------------Perform eigenanalysis of generalized eigenproblem (as in LEM)------------------
if size(T_trainingSet, 1) >= 200 && no_dims < (size(T_trainingSet, 1) / 2) % ��������>200�ҽ�ά�������<��������1/2
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

% Sort eigenvalues in descending order and get smallest eigenvectors
[eigvalue, ind] = sort(diag(eigvalue), 'ascend');
projection = eigvector(:,ind(1:no_dims));

% Compute final linear basis and project data
allprojectionFace = (T_trainingSet * projection)';
%allprojectionFace.M = projection;
%allprojectionFace.mean = mean(training_set, 1);

%------------------����1NN���������з��࣬��ͳ����ȷ��------------------
right = 0;
for x = 1:numTestInstance
    afterProjection = projection' * testingSet(:,x);
    error = zeros(numTrainInstance,1);
    for i=1:numTrainInstance
        %�����ع�ͼ����󵽸������ͼ������ľ���
        dis = afterProjection -allprojectionFace(:,i);
        for j=1:size(dis,2)
            error(i) =error(i)+ norm(dis(:,j));
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
disp('accuracy is');
accuracy
