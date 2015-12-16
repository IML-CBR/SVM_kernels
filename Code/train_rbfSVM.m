function [  model, v ] = train_rbfSVM( labels, data, lambda, sigma )
% Funtion for training a SVM that dose consider errors in the cost
% function, with the dual algorithm solution
    m = size(data,1);
    n = size(data,2);
    X = data';
    Y = labels;
    K = exp( -L2_distance(X,X)/(2*sigma^2));
    tolerance = 1e-4 ;
    cvx_begin
        variables v(m)
        maximise(v'*ones(m,1)-(1/2)*ones(1,m)*(v'*diag(Y)*K*diag(Y)*v)*ones(m,1));%ones -> v
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
    model.sigma = sigma;

if ~ isempty(model.margin)
  % This works almost all times
  model.m = 1-mean(Y(model.margin)'.*(Y(model.margin)' - model.vy(model.svs)' * K(model.svs,model.margin))) ;

else
  % Special cases to deal with the case in which C is very small
  % and there are no support vectors on the margin

  r = 1 - model.vy' * K * diag(Y) ;
  act = ismember(1:m, model.svs) ;
  pos = Y' > 0 ;

  maxb = min([+r(pos & act),  -r(~pos & ~act)]) ;
  minb = max([-r(~pos & act), +r(pos & ~act)]) ;
  if mean(Y(act)) <= -tolerance
    model.m = maxb ;
  elseif mean(Y(act)) > tolerance
    model.m = minb;
  else
    % Any b in the interval is equivalent
    model.m = mean([minb maxb]) ;
  end
end

