function label_onehot = DataOnehot(label)
    label = label(:);
    label_onehot = full(sparse(1:numel(label), label,1)); 
end