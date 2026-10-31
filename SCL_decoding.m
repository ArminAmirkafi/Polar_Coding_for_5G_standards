clc;
clear;
close all;

% SCL Polar encode parameters
n = 10;
N = 2^n;
A = 500;
crcL = 11; % length of CRC
crcg = fliplr([1 1 1 0 0 0 1 0 0 0 0 1]); % CRC polynomial
K = A + crcL; % information bits
rmax = 4; % max received value
maxqr = 31; % max integer received value
nL = 4; % list size
epsilon = 0.5;  % BEC erasure probability
EbNodB = 2.2;
R = A/N;
EbNo = 10^(EbNodB/10);
sigma = sqrt(1/(2*R*EbNo));

% Construct reliability sequence
Z = epsilon;
for stage = 1:n
    Z_minus = 2*Z - Z.^2;
    Z_plus  = Z.^2;
    Z_new = zeros(1, 2*length(Z));
    Z_new(1:2:end) = Z_minus;
    Z_new(2:2:end) = Z_plus;
    Z = Z_new;
end
[~, reliability_sequence] = sort(Z,'ascend');
Q = reliability_sequence - 1;

% Insert information and frozen bits
u = zeros(1, N);
info_pos = Q(1:K) + 1;
frozen = Q(K+1:end);
F = frozen + 1;

% results
disp('Reliability sequence Q:');
disp(Q);
disp('Information positions (MATLAB indexing):');
disp(info_pos);
disp('Input vector u:');
disp(u);

f = @(a,b) (1-2*(a<0)).*(1-2*(b<0)).*min(abs(a),abs(b)); % minsum
g = @(a,b,c) b + (1-2*c).*a;
satx = @(x,th) min(max(x,-th),th); % saturate FP value

Nbiterrs =0; Nblkerrs=0; Nblocks=100;

for blk = 1:Nblocks
    msg = randi([0,1],1,A); % the message
    [quot,rem] = gfdeconv([zeros(1,crcL) fliplr(msg)],crcg); % division by crcg
    msgcrc = [msg fliplr([rem zeros(1,crcL-length(rem))])]; % output of CRC which is K bits
    u(info_pos) = msgcrc;
    
    % Polar encoding (Kronecker multiplication)
    m = 1;
    x = u; 
    for d=n-1:-1:0
        for i = 1:2*m:N 
           a1 = x(i:i+m-1); % first
           a2 = x(i+m:i+2*m-1); % second
           x(i:i+2*m-1) = [mod(a1+a2 ,2), a2];
        end
        m = m*2;
    end
    
    s = 1 - 2*x; % BPSK symbols
    r = s + sigma*randn(1,N); % AWGN channel
    
    % Quantization
    r = satx(r,rmax);
    rq = round(r/rmax*maxqr);
    
    % nL SC decoders
    LLR = zeros(nL,n+1,N); % beliefs in nL decoder
    ucap = zeros(nL,n+1,N); % decisions in nL decoders
    
    % path metrics
    PML=inf*ones(nL,1);
    PML(1)=0;
    node=0;
    depth=0; % root
    done=0;    
    ns=zeros(1,2*N-1); % node state vector
    ucap(:,n+1,node+1) = 0;
    
    LLR(:,1,:) =repmat(rq,nL,1,1); % belief of the root
    
    while done==0
        % leaf
        if depth == n
            DM=LLR(:,n+1,node+1);
            DM=reshape(DM, nL, 1); % Safer than squeeze so i reshaped it 
            
            if any(F==node+1) % frozen node check
                ucap(:,n+1,node+1) = 0; 
                PML=PML+abs(DM).*(DM<0);
            else
                dec=DM <0; 
                PM2=[PML;PML+abs(DM)];
                [PML,pos]=mink(PM2,nL); % pruning not needed process
                
                pos1=pos>nL; % returns true if decision wasn't preferred
                pos(pos1)=pos(pos1)-nL;
                
                dec=dec(pos); % decision of survivors
                dec(pos1)=1-dec(pos1); % flip those to make opposite decision
                
                % update paths
                LLR=LLR(pos,:,:);
                ucap=ucap(pos,:,:); 
                ucap(:,n+1,node+1)=dec;
            end
            
            if node==N-1
                done=1;
            else
                node=floor(node/2);
                depth=depth-1;
            end
            
        else 
            % nonleaf
            npos = (2^depth-1) + node + 1;
            
            if ns(npos)== 0 % left child
                temp = 2^(n-depth);
                Ln=LLR(:,depth+1,temp*node+1:temp*node+temp);
                Ln=reshape(Ln,nL,temp);
                
                a =Ln(:,1:temp/2);
                b=Ln(:,temp/2+1:end);

                temp= temp/2; % go to left child
                node=node*2;
                depth=depth+1;
                
                LLR(:,depth+1,temp*node+1:temp*node+temp) = f(a,b);
                ns(npos)=1;
            
            elseif ns(npos)==1 % right child
                temp= 2^(n-depth);
                Ln=LLR(:, depth+1,temp*node+1:temp*node+temp);
                Ln=reshape(Ln,nL,temp);
                
                a=Ln(:,1:temp/2);
                b=Ln(:,temp/2+1:end);
                 
                % in need of decisions of left child
                left_node =2*node;
                left_depth=depth+1;
                left_temp = temp/2;
                
                ucappp= ucap(:,left_depth+1,left_temp*left_node+1:left_temp*(left_node+1));
                ucappp=reshape(ucappp, nL, left_temp);
                
                node=2*node+ 1;
                depth=depth+1; 
                temp=temp/2;
                
                LLR(:,depth+1,temp*node+1:temp*(node+1))= g(a,b,ucappp);
                ns(npos)=2;

            elseif ns(npos)==2 % lets get to parent
                temp=2^(n-depth);
                lnode=2* node;
                rnode=2*node+1;
                child_depth= depth+1; 
                child_temp=temp/2;
                
                ucapl =ucap(:,child_depth+1,child_temp*lnode+1:child_temp*(lnode+1));
                ucapl=reshape(ucapl,nL,child_temp);
                
                ucapr=ucap(:,child_depth+1,child_temp*rnode+1:child_temp*(rnode+1));
                ucapr=reshape(ucapr,nL, child_temp);
                
                ucap(:,depth+1,temp*node+1:temp*(node+1))=[mod(ucapl+ucapr,2),ucapr];
                
                node=floor(node/2);
                depth=depth- 1;
            end
        end
    end
    
    msg_cap1=ucap(:,n+1,info_pos);
    msg_cap1=reshape(msg_cap1,nL,K);
    
    cout=1;
    for cl=1:nL
        [q1,r1]=gfdeconv(fliplr(msg_cap1(cl,:)),crcg);
        if isequal(r1, 0)
            cout = cl;
            break
        end
    end
    msg_cap = msg_cap1(cout, 1:A);
    
    % errors
    Nerrs= sum(msg ~=msg_cap);
    if Nerrs> 0
        Nbiterrs = Nbiterrs + Nerrs;
        Nblkerrs = Nblkerrs + 1;
    end
            
end

BER_sim = Nbiterrs/A/Nblocks;
FER_sim = Nblkerrs/Nblocks;
fprintf('BER: %f|FER:%f\n',BER_sim,FER_sim);