function [ kfoldIndexes ] = kfoldIndexer( data, k )
    kfoldIndexes = [];
    partialIndexes = (1:1:size(data,1));
    partialK = k;
    for i=1:1:k
        newIndexes = partialIndexes(randperm(size(partialIndexes,2), ... 
        ceil( double(size(partialIndexes,2)/partialK))));
        kfoldIndexes{end+1} = newIndexes;
        partialK = partialK - 1;
        partialIndexes = setdiff(partialIndexes,newIndexes);
    end
end

