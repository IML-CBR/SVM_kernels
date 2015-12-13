function plotRbfSVM( data, labels, model, name )
% Function that plots the SVM model generated for the training data

    Y = labels;
    X = data';
    linewidth = 0.5;


    d = (max(max(data))-min(min(data)))/100; % Step size of the grid
    [x1Grid,x2Grid] = meshgrid(min(data(:,1)):d:max(data(:,1)),...
        min(data(:,2)):d:max(data(:,2)));
    xGrid = [x1Grid(:)';x2Grid(:)'];        % The grid
    K_dense = exp( -L2_distance(data(model.svs,:)',xGrid)/(2*model.sigma^2));
    
    % Predict Y
    y_pred = model.vy(model.svs)' * K_dense;

    
    figure;
    h(1:2) = gscatter(data(:,1),data(:,2),Y,'rb','o+');
    hold on
    
    % Support vectors
    h(3) = plot(data(model.svs,1),...
        data(model.svs,2),'yo','MarkerSize',12, 'linewidth', 0.5);
    hold on;
    
    % Support at margin
    h(4) = plot(data(model.margin,1),...
        data(model.margin,2),'go','MarkerSize',12, 'linewidth', 0.5);
        

    [~,mid]=contour(x1Grid,x2Grid,reshape(y_pred, size(x1Grid,1),size(x1Grid,2)),[0 0]);
    set(mid,'color', 'b', 'linewidth', 1) ;
    [~,marg]=contour(x1Grid,x2Grid,reshape(y_pred, size(x1Grid,1),size(x1Grid,2)),[-model.b -model.b]);
    set(marg,'color', 'r', 'linestyle', '--', 'linewidth', 0.5) ;
    [~,marg]=contour(x1Grid,x2Grid,reshape(y_pred, size(x1Grid,1),size(x1Grid,2)),[model.b model.b]);
    set(marg,'color', 'r', 'linestyle', '--', 'linewidth', 0.5) ;
        % Decision boundary
    title(name)
    legend({'-1','1','Support Vectors','Margin Vectors'},'Location','NorthEastOutside');
    hold off

end

