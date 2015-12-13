function plotRbfSVM_old( data, labels, sigma, name, v, lambda, model )
% Function that plots the SVM model generated for the training data
    X = data';
    Y = labels;
    
    figure;
    linewidth = 0.5;
    range_min = min(data(:,1));
    range_max = max(data(:,1));
    offset = abs(range_max-range_min)/10;
    range = linspace(range_min-offset,range_max+offset,100);
    
    
    
    %Train the SVM Classifier


% % Predict scores over the grid
% d = 0.02;
% [x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
%     min(data3(:,2)):d:max(data3(:,2)));
% xGrid = [x1Grid(:),x2Grid(:)];
% [~,scores] = predict(cl,xGrid);
% 
% % Plot the data and the decision boundary
% figure;
% h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
% hold on
% ezpolar(@(x)1);
% h(3) = plot(data3(cl.IsSupportVector,1),data3(cl.IsSupportVector,2),'ko');
% contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
% legend(h,{'-1','+1','Support Vectors'});
% axis equal
% hold off
    
    
    
%     d = (max(max(data))-min(min(data)))/100;
% [u,t] = meshgrid(min(data(:,1)):d:max(data(:,1)),...
%     min(data(:,2)):d:max(data(:,2)));
% X_dense = [u(:)';t(:)'];
    ur = linspace(range_min,range_max,100) ;
    [u,t] = meshgrid(ur) ;
    X_dense = [u(:)' ; t(:)'] ;
    K_dense = exp( -L2_distance(X(:,model.svs),X_dense)/(2*sigma^2));
%     K_margin = exp( -L2_distance(X(:,model.svs),model.margin)/(2*sigma^2));
    
    f_dense = model.vy(model.svs)' * K_dense + model.b;
    f_dense = reshape(f_dense, size(u,1),size(u,2)) ;

%     cla ;
    imagesc(ur, ur, f_dense) ; colormap HSV ; hold on ;
    [c,hm] = contour(ur, ur, f_dense,[-1 -1]) ;
    set(hm,'color', 'r', 'linestyle', '--') ;
    [c,hp] = contour(ur, ur, f_dense,[+1 +1]) ;
    set(hp,'color', 'g', 'linestyle', '--') ;
    [c,hz] = contour(ur, ur, f_dense,[0 0]) ;
    set(hz,'color', 'b', 'linewidth', 4) ;
    hg  = plot(X(1,Y>0), X(2,Y>0), 'g.', 'markersize', 10) ;
    hr  = plot(X(1,Y<0), X(2,Y<0), 'r.', 'markersize', 10) ;
    hko = plot(X(1,model.svs), X(2,model.svs), 'ko', 'markersize', 15) ;
    hkx = plot(X(1,model.margin), X(2,model.margin), 'kx', 'markersize', 15) ;
    axis tight ;
    legend([hg hr hko hkx hz hp hm], ...
           'pos. vec.', 'neg. vec.', 'supp. vec.', 'margin vec.', ...
           'decision bound.', 'pos. margin', 'neg. margin', ...
           'location', 'northeastoutside') ;
%     switch kernel
%       case 'linear'
%         title(sprintf('linear kernel (C = %g)', C)) ;
%       case 'rbf'
        title(sprintf('RBF kernel (\\lambda = %g, \\sigma = %g)', lambda, sigma)) ;


        
        
% figure;

d = (max(max(data))-min(min(data)))/100; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(min(data(:,1)):d:max(data(:,1)),...
    min(data(:,2)):d:max(data(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];        % The grid
K_dense1 = exp( -L2_distance(data(model.svs,:)',xGrid')/(2*sigma^2));
y_pred = model.vy(model.svs)' * K_dense1 + model.b;
% predict
figure;
h(1:2) = gscatter(data(:,1),data(:,2),Y);
hold on
h(3) = plot(data(model.svs,1),...
    data(model.svs,2),'ko','MarkerSize',10);
    % Support vectors
contour(x1Grid,x2Grid,reshape(y_pred, size(x1Grid,1),size(x1Grid,2)),[0 0]);
    % Decision boundary
title('Scatter Diagram with the Decision Boundary')
legend({'-1','1','Support Vectors'},'Location','Best');
hold off





% 
% 
% 
%     SVsIndexes = find((t > 0)&(t < lambda));
%     SVs = data(SVsIndexes, :);
%     SVlabels = labels(SVsIndexes);
%     d = (SVlabels'.*(model'*SVs'));
% 
%     figure;
%     linewidth = 0.5;
%     range_min = min(data(:,1));
%     range_max = max(data(:,1));
%     offset = abs(range_max-range_min)/10;
%     range = linspace(range_min-offset,range_max+offset,100);
%     w_lim = (model(1)*range)/(-model(2));
% %     marg1 = (model(1)*range + mean(d))/(-model(2));
% %     marg2 = (model(1)*range - mean(d))/(-model(2));
% %     
%     positive = data(find(labels > 0),:);
%     negative = data(find(labels < 0),:);
%     graph = plot(positive(:,1),positive(:,2), '+', negative(:,1), negative(:,2), 'o');
%     set(graph(1),'LineWidth',linewidth);
%     set(graph(1),'MarkerFaceColor',[0 0 0.5]);
%     set(graph(2),'LineWidth',linewidth);
%     set(graph(2),'Color',[0.5 0 0]);
%     hold on;
%     
%     % Distance is calculated as r = (w'*X_i + b)
%     distances = (data*model).*labels;
% %     supIndex = intersect(find(v < lambda),find(v > 0));
%     supIndex = (unique([find(distances==d(1)),find(distances==d(2)),find(distances==d(3))]'))';
%     suports = data(supIndex,:);
% %     while length(suports)>3
% %         decimals = decimals + 1;
% %         suports = data(find(arrayfun(@(x) roundx(x,decimals,'round'),(distances))==d),:);
% %     end
% %     errIndex = find(v == lambda);
%     errIndex = intersect(intersect(find(distances < d(1)),find(distances < d(2))),find(distances < d(3)));
%     errors = data(errIndex,:);
%     
%     
%     scatter(errors(:,1),errors(:,2),200,'y','o','LineWidth',1.5);
%     hold on
%     scatter(suports(:,1),suports(:,2),200,'g','o','LineWidth',1.5);
%     
%     
%     hold on;
%     plot(range,w_lim, '-b');%, range,marg1, '--r', range,marg2, '--r');
%     axis tight
%     title(name)
%     
%     xlabel('x1')
%     ylabel('x2')
%     hold off
    
%Real distance is not used
%     suports = data(find(arrayfun(@(x) roundx(x,5,'round'),...
%         ((abs([data,ones(size(data,1),1)]*model)/...
%         ((model(1).^2+model(2).^2).^0.5))))<=1),:);

end

