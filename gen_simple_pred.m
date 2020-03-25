%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This part of the code was used to generate the result of examples 1 and 2. It generates the phase space, and make predictions
% Please consider citing the paper if you find this code useful:
% "Khan and Takehisa (2020). Diagnosing intermittent faults through non-linear analysis. IFAC2020"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%data1=xd;

data1=tr2(1000:end,1); % input 1-d data
[n_1,m_1]=size(data1); %m_1 is no of columns, n_1 no of rows

%% n-d phase space conversion

N = length(data1); tau =5; m1=3; % choose appropriate tau and embedding dimension using autocorrelation and FNN
N2 = N - tau * (m1 - 1); % recalculate length with tau delay

temp=zeros(N2,m_1*m1);
xe=zeros(N2,m1-1); 

for t_i=1:m_1
    Y_1=data1(:,t_i);
    for mi = 1:m1
        for k=1:N2
            xe(k, mi) = Y_1(k + tau*(mi-1));
        end
    end
    temp(:,m1*t_i-(m1-1):m1*t_i)=xe(:,:);
end

data1=temp; % store of future use

%% prediction

m=0;n=0;intial_value=0; 

% plot(data1(:,1)) 

m=length(data1); 
n=m1;
dim=m1; %dim=n;
steps=50; % how many steps wanted for prediction
t_past=50; % how many steps in the past for overlap

avg=zeros(steps,dim); % initialise the reponse matrix
i3=1; % this will be used for global column assignment

for i2=1:m_1 %(global)

    intial_value(1,1:dim)=data1(m-t_past,i3:i3+m1-1); % pick a starting point
    temp=0; % initialise temporary variable; n is the no of columns
    k_nn=n+1; % KNN has to be greater or equal to 2; rule of thumb

%------ being prediction process
    for i=1:steps % how many steps
        [cIdx,cD] = knnsearch(data1(1:m-t_past,i3:i3+m1-1),intial_value(1,1:dim),'K',k_nn,'Distance','euclidean');

        for j =2:k_nn % look at the future values of past responses
          
            if cIdx(1,j)==length(data1(:,i3:i3+m1-1)) % incase the nearest neighbour has no future value
                cIdx(1,j)=cIdx(1,j-1); % use the j-1 value
            end
        %avg(i,:)=avg(i,:)+X3(cIdx(1,j)+1,:);      
        temp=temp+data1(cIdx(1,j)+1,i3:i3+m1-1); % keep adding the nearest neighbour data points   

        end

    avg(i,i3:i3+m1-1)=temp/(k_nn-1); % take the average of the nearest neighbour data points   
    intial_value(1,1:dim)=avg(i,i3:i3+m1-1); % intialise next value
    %data1(i+m,1:dim)=intial_value(1,1:dim);
    temp=0; % rest variable 
    end

    i3=i3+m1; % increment global assignment
end

%%%%%%%%%%%%%%%%%%%%%
%% plot results
hold on
plot(m-t_past+1:m-t_past+steps,avg(:,1)) 
line([m-t_past+1 m-t_past+1], [-1.5 1.5]);
