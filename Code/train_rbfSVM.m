function [  model, v, K ] = train_rbfSVM( labels, data, lambda, sigma )
% Funtion for training a SVM that dose consider errors in the cost
% function, with the dual algorithm solution
    m = size(data,1);
    n = size(data,2);%+1;
    X = data';%,ones(m,1)]';
    Y = labels;
    K = exp( -L2_distance(X,X)/(2*sigma^2));
%     Y = diag(y) ;
    tolerance = 1e-4 ;
%     figure; imagesc(K); title('Gram matrix for sigma = 1');
    cvx_begin
        variables v(m)
%         r = (v'*ones(m,1)-(1/2)*sum(sum((v'.*Y')*(v.*Y)*X'*X)))
        maximise(v'*ones(m,1)-(1/2)*ones(1,m)*(v'*diag(Y)*K*diag(Y)*v)*ones(m,1));%ones -> v
%         maximize(v'*ones(m,1)-(1/2)*ones(1,m)*((v'.*Y')*(v.*Y)*(X')*X)*ones(m,1));
        subject to
            v'*Y == 0;
            lambda >= v;
            v >= 0;
    cvx_end
    
    % Fix v values
%     finish = 0;
%     decimals = 0;
%     while (decimals < 6) && ~finish %&& (lambda/10^decimals > mean(v)/100)
%         finish = (size(Y,1) - (sum(v <= lambda/10^decimals)+sum((arrayfun(@(x) roundx(x,decimals,'round'),v) == lambda))) == 3);
%         if ~finish decimals = decimals+1; end
%     end
    
    
%     v(find(v <= lambda/10^(decimals))) = 0;
%     v(find((arrayfun(@(x) roundx(x,decimals,'round'),v)) == lambda)) = lambda;
%     w = (v(find(v>0))'.*Y(find(v>0))'*K(find(v>0),:))';
    
    
    
    model.margin = find(v > tolerance * lambda & v < (1 - tolerance) * lambda) ;
    model.svs = find(v > tolerance * lambda) ;
    model.v = v ;
    model.vy = diag(Y) * v ;
    model.K = K;

% if ~ isempty(model.margin)
%   % This worsk 99% of the times
%   model.b = mean(Y(model.margin)' - model.vy' * K(:,model.margin)) ;
% 
% else
%   % Special cases to deal with the case in which C is very small
%   % and there are no support vectors on the margin
% 
%   r = 1 - model.vy' * K * diag(Y) ;
%   act = ismember(1:n, model.svs) ;
%   pos = Y > 0 ;
% 
%   maxb = min([+r(pos & act),  -r(~pos & ~act)]) ;
%   minb = max([-r(~pos & act), +r(pos & ~act)]) ;
%   if mean(Y(act)) <= -tolerance
%     model.b = maxb ;
%   elseif mean(Y(act)) > tolerance
%     model.b = minb;
%   else
%     % Any b in the interval is equivalent
%     model.b = mean([minb maxb]) ;
%   end
    


    %     yi(a'xi -b) = d
    % yi(w'xi -b) = yj(w'xj -b)
    % yi/yj = 1 if equals
    % w'xi -b = w'xj -b
   
    
%     auxB = 0;
%     for i=1:1:size(SVs,1)
%         auxB = auxB + 
%     end
%     b = auxB/SVcnt;
% (cat(2,SVs,-SVlabels)*[w;1])
% SVs(:,3)=-SVlabels
% v'*ones(m,1)-(1/2)*ones(1,m)*((v'.*Y')*(v.*Y)*(X')*X)*ones(m,1)
% v'*ones(m,1)-(1/2)*ones(1,m)*(v'*diag(Y)*X'*X*diag(Y)*v)*ones(m,1)
% b = 0;
%     for i = 1:size(SVlabels,1)
%         b = -(SVs(i,:)*w - SVlabels(i)) + b;
%     end
%     b = b/size(SVlabels,1);
% sum(((w'*SVs')'-SVlabels))/3
%     b = mean([SVs,-SVlabels]*[w;1],1);
    
%     a'xi - yi = b
% decimals = 5;
%     sum((arrayfun(@(x) roundx(x,decimals,'round'),v)) == lambda)
%     w(end+1) = b;
end

