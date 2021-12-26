clear all
n = 5; % dimension of the process
p = 1;  % order of the AR model
% length of the data
mo=10;
N=100;
A=toeplitz([0.5 0.2 zeros(1,n-2)])
A(1:4,1:4)= [0.6 0.2 0.4 0.5;
    0 0.7 0.5 0;
    0 0 0.5 -0.2;
    0 0  0 0.7 ];

Rv=eye(n);


v=mvnrnd(zeros(n,1),Rv,N)';
y=zeros(n,N);
for i =1: N-1
    y(:,i+1)= A* y(:,i)+ v(:,i);
end

Sigma_est= sTop(y',p);
gamma=exp(-6:2:2);
ng=length(gamma);
A1store=zeros(n,n,ng);
for gi=1:ng
    cvx_begin sdp
    variable X(n*(p+1),n*(p+1)) symmetric;
    variable Y
    expression D(n,n,p+1);
    expression h;
    minimize(-log_det(X(1:n,1:n))+trace(Sigma_est'*X) + gamma(gi)*Y);
    subject to
    for l=0:p
        D(:,:,1)=D(:,:,1)+X(((l)*n+1):((l+1)*n),((l)*n+1):((l+1)*n));
    end
    %Dj for j in [1,p]
    for m=1:p
        for l=0:p-m
            D(:,:,m+1) = D(:,:,m+1) + 2*X(((l)*n+1):((l+1)*n),((l+m)*n+1):((l+m+1)*n));
        end
    end
    
    for i=1:n
        for j= i+1:n
            h=h+max(max(abs(D(j,i,1)),abs(D(j,i,2))),abs(D(i,j,2)));
        end
    end
    Y>=h;
    X>=0;
    cvx_end
%% obtain the lasso estimate
    [BT,L]=svd(X);
    B=(BT*L^(1/2))';
    B= B(1:n,:);
    B'*B-X
    B0=B(:,1:n);
    B1=B(:,n+1:end);
    Sigma = (B0'*B0)^(-1)
    L_sig= chol(Sigma)
    L_sig'*L_sig
    A1=-B0^(-1)*B1;
    A1store(:,:,gi)=A1;
end

%% LS estimate
z= iddata(y(1,:)',y(2:n,:)')
nb=mo*ones(1,n-1);
nk=ones(1,n-1);
[Nd, ~, Nu, Ne] = size(z);
ind = cumsum(nb+1-nk); ind = [0 ind]; % index of the estimated FIR parameters for each of Nu blocks
Nth = ind(end);
Rt = zeros(0,Nth+1);
yy = pvget(z,'OutputData');
uu = pvget(z,'InputData');

Ney = sum(Nd-max(nb));
yt = zeros(Ney,1);
Phi = zeros(Nth,Ney);
nt = cumsum(Nd-max(nb)); nt = [0 nt];

for ni = 1:Nu
    for ne = 1:Ne
        for nj = 1:Nd(ne)-max(nb)
            Phi(ind(ni)+1:ind(ni+1),nj+nt(ne)) = flipud(uu{ne}(nj+max(nb)-nb(ni):(nj+max(nb)-nk(ni)),ni));
        end
    end
end
for ne = 1:Ne
    yt(nt(ne)+1:nt(ne+1)) = yy{ne}(max(nb)+1:end);
end
% Calculate the QR factor of Phi and yt
Rt = triu(qr([Phi' yt]));
try
    Rt = Rt(1:Nth+1,:);
catch
    Rt = [Phi' yt];
end
% Calculate the variance of the measurement noise
X_ls = (Rt(:,1:Nth)'*Rt(:,1:Nth))\Rt(:,1:Nth)'*Rt(:,end);
sigma_ls = sum((Rt(:,end)-Rt(:,1:Nth)*X_ls).^2)/(Ney-Nth);
X_ls= reshape(X_ls, mo,n-1);

%%    kernel estiamte % it is not a open sourse, but you can use the equivalent matlab function impulsest(), designed by Chen(tschen@cuhk.edu.cn)

% % % % load(data_path)
% % % % data = Data{1};
% % % path = 'D:\Course\Graphical model\project\Rels';
% % % if ~exist(path)
% % %     path = '/home/biqiang/Rels';
% % % end
% % % addpath([path '/Tuning_methods'])
% % % % addpath([path '/Rels/Tuning_methods_old']);
% % % % addpath(genpath([path '/Rels']));
% % % addpath([path '/Basic_functions']);
% % % kernel = 'TC';
% % % np = mo;
% % % % [M_Sy estin_Sy] = rfir_surey(data,[np, 1],kernel);
% % % % [M_NEW estin_NEW] = rfir_new(data,[np, 1],kernel);
% % % % Methods = {'OG','OGP1','OGP2','MSEG', 'SG', 'OY', 'MSEY', 'SY','GCV','LOOCV'};
% % % data=z
% % %
% % % [M_EB estin_EB] = rfir_eb(data,[np, 1],kernel);
% % % theta_EB = M_EB.b(1:end);
% % % X_ls_tc=zeros(mo,n-1);
% % % for i =1:n-1
% % %     X_ls_tc(:,i)=theta_EB{i}(2:end)';
% % % end
% % %
% % %
% % %

%% plot result
v=mvnrnd(zeros(n,1),Rv,N)';
y=zeros(n,N);
for i =1: N-1
    y(:,i+1)= A* y(:,i)+ v(:,i);
end
y1_hat = zeros(ng, N);
for i= mo+1:N
    for p =1:mo
        for ig=1:ng
            y1_hat(ig,i) =  y1_hat(ig,i)+A1(1,1)^p *A1store(1,2:n,ig)*y(2:n,i-p);
        end
    end
end
y1_hat_ls = zeros(1, N);
for i= mo+1:N
    for p =1:mo
        y1_hat_ls(1,i) =  y1_hat_ls(1,i)+X_ls(p,:) *y(2:n,i-p);
    end
end
% y1_hat_tc = zeros(1, N);
% for i= mo+1:N
%     for p =1:mo
%         y1_hat_tc(1,i) =  y1_hat_tc(1,i)+X_ls_tc(p,:) *y(2:n,i-p);
%     end
% end

subplot(311)
plot(1:N, y(1,:)-[0 v(1,1:end-1)],'k--',1:N,y1_hat_ls(1,:))
title('ML estimate (n=5), MSE=2.29')
subplot(312)
plot(1:N, y(1,:)-[0 v(1,1:end-1)],'k--')
hold on
for ig=1:ng
    plot(1:N,y1_hat(ig,:))
end
hold off
xlabel('time(t)')
legend('true ground','\gamma=exp(-6)','\gamma=exp(-4)','\gamma=exp(-2)', '\gamma=1', '\gamma=exp(2)')
title('MEs estimate (n=5),MSE(best)=1.34')
%
% subplot(313)
% plot(1:N, y(1,:)-[0 v(1,1:end-1)],'k--',1:N,y1_hat_tc(1,:))
% title('MEt estimate (n=5),MSE=1.12')

zz=y(1,:)-[0 v(1,1:end-1)]-y1_hat(2,:);
mse(zz(10:end))
zz=y(1,:)-[0 v(1,1:end-1)]-y1_hat(3,:);
mse(zz(10:end))
zz=y(1,:)-[0 v(1,1:end-1)]-y1_hat(4,:);
mse(zz(10:end))

zz=y(1,:)-[0 v(1,1:end-1)]-y1_hat_ls(1,:);
mse(zz(10:end))
% zz=y(1,:)-[0 v(1,1:end-1)]-y1_hat_tc(1,:);
% mse(zz(10:end))

%% these functions credit to Zorzi (https://github.com/CyclotronResearchCentre/SparseLowRankIdentification)
function D = get_D_short(X, n, p)
% Function to compute the adjoint of T with output in 3-d array
% p is the AR order
% n is the dimension of the process
D = zeros(n,n,(p+1));
%D0
for l=0:p
    D(:,:,1)=D(:,:,1)+X(((l)*n+1):((l+1)*n),((l)*n+1):((l+1)*n));
end
%Dj for j in [1,p]
for m=1:p
    for l=0:p-m
        D(:,:,m+1) = D(:,:,m+1) + 2*X(((l)*n+1):((l+1)*n),((l+m)*n+1):((l+m+1)*n));
    end
end
end

function C=sTop(X,p)
% Computes the sample Toeplitz covariance matrix C of order p
% from the given data X (rows is time, columns are the component)
%
% Note that C has dimension n(p+1) x n(p+1) s.t.
%
%       C=[C0  C1 .... Cn
%          C1' C0 .... :
%          :           :
%          Cn'          ]
%
% where C_k=E[x(t+k)x(t)^T]

n=size(X,2);
N=size(X,1);
C_line=zeros(n,n,2*p+1);
for k=p+1:2*p+1
    for t=1:N-k+p+1
        C_line(:,:,k)=C_line(:,:,k)+N^-1*X(t+k-p-1,:)'*X(t,:);
    end
end
for k=1:p
    C_line(:,:,k)=C_line(:,:,-k+2*p+2)';
end



C = zeros(n*(p+1));
for i=1:p+1
    for j=1:p+1
        C(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=C_line(:,:,(j-i)+p+1);
    end
end

end
