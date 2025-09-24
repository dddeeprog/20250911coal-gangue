%{
type -
linear : k(x,y) = x'*y
polynomial : k(x,y) = (γ*x'*y+c)^d
gaussian : k(x,y) = exp(-γ*||x-y||^2)
sigmoid : k(x,y) = tanh(γ*x'*y+c)
laplacian : k(x,y) = exp(-γ*||x-y||)
kernel = Kernel('type', 'gaussian', 'gamma', value);
kernel = Kernel('type', 'polynomial', 'degree', value);
kernel = Kernel('type', 'linear');
kernel = Kernel('type', 'sigmoid', 'gamma', value);
kernel = Kernel('type', 'laplacian', 'gamma', value);
degree - d
offset - c
gamma - γ
%}
function mappingData = FUN_KPCA(data,n)
    kernel = Kernel('type', 'gaussian', 'gamma', 2);
    parameter = struct('numComponents',n,'kernelFunc', kernel);
    % build a KPCA object
    kpca = KernelPCA(parameter);
    % train KPCA model
    kpca.train(data);
    %　mapping data
    mappingData = kpca.score;
end





