function [W_slpm] = ct_SLPM(data,gnd,varargin)
% SLPM : Soft Locality Preserving Maps
%
%       [W_slpm] = ct_SLPM(data,gnd,varargin)
% 
%             Input:
%               data    - Data matrix. Each row vector of fea is a data point.
%
%               gnd     - Label vector
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%                 intraK         = 0  
%                                     Sc:
%                                       Put an edge between two nodes if and
%                                       only if they belong to same class. 
%                                > 0  Sc:
%                                       Put an edge between two nodes if
%                                       they belong to same class and they
%                                       are among the intraK nearst neighbors of
%                                       each other in this class.  
%                 interK         = 0  Sp:
%                                       Put an edge between two nodes if and
%                                       only if they belong to different classes. 
%                                > 0
%                                     Sp:
%                                       Put an edge between two nodes if
%                                       they rank top interK pairs of all the
%                                       distance pair of samples belong to
%                                       different classes
%				  dim         	 - required output dimensionality
%
%				  beta         	 - required output dimensionality
%                                       
%
%             Output:
%               W_slpm 	  - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*W_slpm
%                           will be the embedding result of x.
% 
%
%    Examples:
%
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.intraK = 5;
%       options.interK = 40;
%       [W_slpm] = ct_SLPM(fea,gnd, options);
%       Y = fea*W_slpm;
% 
%
%	Reference:
%
%   C. Turan, K.M. Lam. "Soft Locality Preserving Map for Facial Expression
%    Recognition." submitted to Pattern Recognition. 
%
%   version 1.0 -- Jun/2017
%
%   Written by Cigdem TURAN (cigdem.turan AT connect.polyu.hk)
%
	if nargin == 3
		options = varargin{1};
		if ~isfield(options,'interK')
			options.interK = 25;
		end
		if ~isfield(options,'intraK')
			options.intraK = 10;
		end
		if ~isfield(options,'dim')
			options.dim = 11;
		end
		if ~isfield(options,'beta')
			options.beta = 3;
		end
		if ~isfield(options,'t')
			options.t = 1;
		end
	else
		options = struct;
		options.interK = 25;
		options.intraK = 10;
		options.dim = 11;
		options.beta = 3;
		options.t = 1;
	end

    Label = unique(gnd);
    nLabel = length(Label);
    nSmp = size(data,1);

    D = EuDist2(data,[],0);
    nIntraPair = 0;
    if intraK > 0
        G = zeros(nSmp*(intraK+1),3);
        idNow = 0;
        for i = 1:nLabel
            classIdx = find(gnd==Label(i));
            DClass = D(classIdx,classIdx);
            [dump, idx] = sort(DClass,2); % sort each row
            clear DClass;
            nClassNow = length(classIdx);
            nIntraPair = nIntraPair + nClassNow^2;
            if intraK < nClassNow
                idx = idx(:,1:intraK+1);
                dump = dump(:,1:intraK+1);
            else
                idx = [idx repmat(idx(:,end),1,intraK+1-nClassNow)];
                dump = [dump repmat(dump(:,end),1,intraK+1-nClassNow)];
            end

            dump = exp(-dump/(2*options.t^2));

            nSmpClass = length(classIdx)*(intraK+1);
            G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[intraK+1,1]);
            G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
%             G(idNow+1:nSmpClass+idNow,3) = dump(:);
            G(idNow+1:nSmpClass+idNow,3) = 1;

            idNow = idNow+nSmpClass;
            clear idx dump; 
        end
        Sc = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
        [I,J,V] = find(Sc);
        Sc = sparse(I,J,1,nSmp,nSmp);
        Sc = max(Sc,Sc');
        clear G
    else
        Sc = zeros(nSmp,nSmp);
        for i=1:nLabel
            classIdx = find(gnd==Label(i));
            nClassNow = length(classIdx);
            nIntraPair = nIntraPair + nClassNow^2;
            Sc(classIdx,classIdx) = 1;
        end
    end

    if interK > 0 && (interK < (nSmp^2 - nIntraPair))
        maxD = max(max(D))+100;
        for i=1:nLabel
            classIdx = find(gnd==Label(i));
            D(classIdx,classIdx) = maxD;
        end

        [dump,idx] = sort(D,2);
        dump = dump(:,1:interK);
        dump = exp(-dump/(2*options.t^2));
        idx = idx(:,1:interK);
        wholeIdx = [1:nSmp]';
        G = zeros(nSmp*(interK),3);
        G(:,1) = repmat(wholeIdx,[interK,1]);
        G(:,2) = wholeIdx(idx(:));
        G(:,3) = dump(:);

        Sp = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
        Sp = sparse(max(Sp,Sp'));
    else
        display('here');
        Sp = ones(nSmp,nSmp);
        for i=1:nLabel
            classIdx = find(gnd==Label(i));
            Sp(classIdx,classIdx) = 0;
        end
    end

    Dp = full(sum(Sp,2));
    Sp = -Sp;
    for i=1:size(Sp,1)
        Sp(i,i) = Sp(i,i) + Dp(i);
    end

    Dc = full(sum(Sc,2));
    Sc = -Sc;
    for i=1:size(Sc,1)
        Sc(i,i) = Sc(i,i) + Dc(i);
    end

    W = Sp - beta*Sc;
    WPrime = data'*W*data;
    WPrime = max(WPrime,WPrime');
    
    [eigvec,eigval] = eig(WPrime);
    [~, idx] = sort(diag(eigval),'descend');
    W_slpm = eigvec(:,idx(1:dim));
    
end