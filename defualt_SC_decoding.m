clc;
clear;
close all;

% Polar code parameters(change it anytime)
n = 4;
N = 2^n;
K = 10; % information bits
epsilon = 0.5;  % BEC erasure probability
EbNodB = 4;
R = K/N;
EbNo = 10^(EbNodB/10);
sigma = sqrt(1/(2*R*EbNo));

%% Step 1: Construct reliability sequence
Z = epsilon;
for stage = 1:n
    Z_minus = 2*Z - Z.^2;
    Z_plus  = Z.^2;
    Z_new = zeros(1, 2*length(Z));
    Z_new(1:2:end) = Z_minus;
    Z_new(2:2:end) = Z_plus;
    Z = Z_new;
end

[~, reliability_sequence] = sort(Z, 'ascend');
Q = reliability_sequence - 1;

%% Step 2: Generate information bits
c = randi([0,1],1,K);

%% Step 3: Insert information and frozen bits
u = zeros(1, N);
info_pos = Q(1:K) + 1;
frozen = Q(K+1:end);
u(info_pos) = c;

%% Polar encoding (Kronecker multiplication)
m = 1;
x = u; 
for d = n-1:-1:0
    for i = 1:2*m:N 
       a1 = x(i:i+m-1); % first
       a2 = x(i+m:i+2*m-1); % second
       x(i:i+2*m-1) = [mod(a1+a2 , 2), a2];
    end
    m = m*2;
end

s = 1 - 2*x; % BPSK symbols
r = s + sigma*randn(1,N); % AWGN channel

%% results
disp('Reliability sequence Q:');
disp(Q);
disp('Information positions (MATLAB indexing):');
disp(info_pos);
disp('Input vector u:');
disp(u);
disp('Encoded codeword x:');
disp(x);

% beliefs
L = zeros(n+1,N);
L(1,:) = 2 * r / sigma^2; 

node = 0;
depth = 0;
done = 0;
uhat = zeros(1,N);

% save sums
SUMS = zeros(n+1,N);
ns = zeros(1,2*N-1); % save for 2*N-1 nodes

f = @(a,b) (1-2*(a<0)).*(1-2*(b<0)).*min(abs(a),abs(b)); % min sum
g = @(a,b,c) b + (1-2*c).*a;

is_frozen = false(1, N);
is_frozen(frozen + 1) = true;

while done == 0
    temp = N/2^depth; % step for node traversal
    npos = 2^depth + node;
    startIdx = temp*node + 1;
    endIdx = temp*node + temp;

    % leaves
    if depth == n
        idx = node + 1;
        if is_frozen(idx)
            uhat(idx) = 0;
        else
            uhat(idx) = double(L(depth+1,idx) < 0);
        end
        SUMS(depth+1,idx) = uhat(idx);

        if depth == 0
            done = 1;
        else
            % go upper node
            node = floor(node/2);
            depth = depth - 1;
        end
         
    % no leaves
    else
        L0 = L(depth+1, startIdx:endIdx);
        a = L0(1:temp/2);
        b = L0(temp/2+1:end);

        if ns(npos) == 0
            % go to left child
            LLR_l = f(a,b);
            sub_start = startIdx;
            sub_end = startIdx + temp/2 - 1;
            L(depth+2, sub_start:sub_end) = LLR_l;
            node = 2*node;
            depth = depth + 1;
            ns(npos) = 1;

        elseif ns(npos) == 1
            % go to right child
            sub_start_left = startIdx;
            sub_end_left = startIdx + temp/2 - 1;
            c = SUMS(depth+2, sub_start_left:sub_end_left);
            
            LLR_r = g(a,b,c);
            sub_start_right = startIdx + temp/2;
            sub_end_right = endIdx;
            L(depth+2, sub_start_right:sub_end_right) = LLR_r;

            ns(npos) = 2;
            node = 2*node + 1;
            depth = depth + 1;
            
        elseif ns(npos) == 2
            sub_start_left = startIdx;
            sub_end_left = startIdx + temp/2 - 1;
            sub_start_right = startIdx + temp/2;
            sub_end_right = endIdx;

            left_part = SUMS(depth+2, sub_start_left:sub_end_left);
            right_part = SUMS(depth+2, sub_start_right:sub_end_right);
            SUMS(depth+1, startIdx:endIdx) = [mod(left_part+right_part,2), right_part];
            
            if depth == 0
                done = 1;
            else
                node = floor(node/2);
                depth = depth - 1;
            end
        end
    end
end

disp('Decoded uhat:');
disp(uhat);
disp('Number of bit errors:');
disp(sum(uhat~=u));