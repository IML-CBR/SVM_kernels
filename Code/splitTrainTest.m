function [ train_data, test_data ] = splitTrainTest(data, percentage_train)
    num_total_instances = size(data,1);    
    num_train_instances = round(num_total_instances*percentage_train);

    train_indexes = randperm(num_total_instances, num_train_instances);
    train_data = data(train_indexes,:);
    
    test_indexes = setdiff(1:num_total_instances, train_indexes);
    test_data = data(test_indexes,:);
end