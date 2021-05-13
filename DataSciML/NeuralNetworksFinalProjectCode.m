%Nathan Lutes
%Project RBF test
%4/25/2019
%%% This code implements a radial basis function neural network to classify
%%% chess boards based on what pieces are present on the board and where
%%% they are on the board.
%%% Input: a jpeg image of a chess board
%%% Output: notation of the chess board
%%
clear
clc
try
    close all
catch
end
%%
%Load Data from dataset
%get folder directory
%home computer
direct='C:\Users\Nathan\Desktop\School Spr. 2019\NeuralNets\Project\Pictures';
%school computer
%direct='\\minerfiles.mst.edu\dfs\users\nalmrb\Desktop\Neural Networks\Project\test';
%create a structure with names of files
S=dir(fullfile(direct));
%get rid of first two fields because unimportant
S(1:2)=[];
%define number of images to import
num_image=1500; %length(S)
%storage
datasetfull=[];
%loop that reads in each image
for i=1:num_image  
    filename=S(i).name;
    %housekeeping
    if length(filename)<=3
        continue
    else
        %concatenate strings
        str1=strcat(direct,'\\',S(i).name);
        %read in image
        A=imread(str1);
        %create grayscale of image
        gray_A=rgb2gray(A);
        %resize image
        %gray_A_r=imresize(gray_A,1/2);
        %discretize image into board squares and store as vectors
        dataset=[];
        for j=1:8   %columns
            for k=1:8   %rows
                square=gray_A(1+50*(k-1):50+50*(k-1),...
                    1+50*(j-1):50+50*(j-1));
                square=reshape(square,[1,50^2]);
                dataset=[dataset; square];
            end
        end
        %normalize data
        dataset=double(dataset);
        dataset=normalize(dataset);
        %pca dimension reduction
        coeff=pca(dataset);
        data_red=dataset*coeff(:,1:15);
        datasetfull(:,:,i)=data_red;
    end
end
%%
%create target vectors from labels
%create keyset and valueset
keyset={'K', 'Q', 'B', 'N', 'R', 'P', 'k', 'q', 'b', 'n', 'r', 'p'};
valueset=1:12;
%create mapping object
map=containers.Map(keyset,valueset);
%get dimensions of datasetfull
[r,c,d]=size(datasetfull);
for i=1:d
    %create 64x64 matrix
    board=zeros(8,8);
    %create "board"
    r=1;
    c=1;
    filename=S(i).name;
    label=filename(1:end-5);
    labels{i}=label;
    for j=1:length(filename)-5
        char=filename(j);
        %determine if char is a number
        number=str2num(char);
        if isempty(number)==0
            c=c+number;
        elseif char=='-'
            r=r+1;
            c=1;
        else
            board(r,c)=map(char);
            c=c+1;
        end
    end 
    boards{i,1}=num2cell(board);
    %create one-hot targets for each square
    for k=1:numel(board)
        y=zeros(1,13);
        y(1,board(k)+1)=1;
        data_w_t(k,:,i)=cat(2,datasetfull(k,:,i),y);
    end  
end
clear S
%%
%create training, validation and test sets
%set number of boards to use as training data
trn_brd=round(0.75*num_image);
%pick random boards and assemble them into one matrix
agg_data=[];
rnd_brd=randperm(d,trn_brd);
for i = 1:length(rnd_brd)
    agg_data=[agg_data; data_w_t(:,:,rnd_brd(i))];
end
agg_data=agg_data';
[s1, s2]=size(agg_data);
%%
%resample dataset to get a balanced dataset
blank=[];
King=[];
Queen=[];
Bishop=[];
Knight=[];
Rook=[];
Pawn=[];
king=[];
queen=[];
bishop=[];
knight=[];
rook=[];
pawn=[];
count=zeros(1,13);
%count number of each class
for i = 1:s2
    if agg_data(s1-12,i)==1
        count(1,1)=count(1,1)+1;
        blank=[blank i];
    elseif agg_data(s1-11,i)==1
        count(1,2)=count(1,2)+1;
        King=[King i];
    elseif agg_data(s1-10,i)==1
        count(1,3)=count(1,3)+1;
        Queen=[Queen i];
    elseif agg_data(s1-9,i)==1
        count(1,4)=count(1,4)+1;
        Bishop=[Bishop i];
    elseif agg_data(s1-8,i)==1
        count(1,5)=count(1,5)+1;
        Knight=[Knight i];
    elseif agg_data(s1-7,i)==1
        count(1,6)=count(1,6)+1;
        Rook=[Rook i];
    elseif agg_data(s1-6,i)==1
        count(1,7)=count(1,7)+1;
        Pawn=[Pawn i];
    elseif agg_data(s1-5,i)==1
        count(1,8)=count(1,8)+1;
        king=[king i];
    elseif agg_data(s1-4,i)==1
        count(1,9)=count(1,9)+1;
        queen=[queen i];
    elseif agg_data(s1-3,i)==1
        count(1,10)=count(1,10)+1;
        bishop=[bishop i];
    elseif agg_data(s1-2,i)==1
        count(1,11)=count(1,11)+1;
        knight=[knight i];
    elseif agg_data(s1-1,i)==1
        count(1,12)=count(1,12)+1;
        rook=[rook i];
    else
        count(1,13)=count(1,13)+1;
        pawn=[pawn i];
    end
end
min_n=min(count);
class={blank; King; Queen; Bishop; Knight; Rook; Pawn; king; queen; bishop;
    knight; rook; pawn};
%resample
NN_data=[];
for i = 1:length(class)
    dummy=class{i};
    dummy=dummy(randperm(length(dummy),min_n));
    NN_data=[NN_data agg_data(:,dummy)];
end
NN_data=NN_data(:,randperm(length(NN_data)));
%%
%design and train an RBF neural nets to classify pieces
mse_goal=0.015;
spread=0.001;
index=0;
NN1=newrb(NN_data(1:s1-13,:),NN_data(s1-12:s1,:),mse_goal,spread);
%test network
y_NN1=NN1(NN_data(1:s1-13,:));
%set threshold to 0.5
yr_NN1=round(y_NN1);
%calculate classification accuracy
n_correct=0;
for i=1:length(yr_NN1)
    if yr_NN1(:,i)==NN_data(s1-12:s1,i)
        n_correct=n_correct+1;
    else
        continue
    end
end
per_correct=n_correct/i * 100;
%%
%Board Classification
[r c z]=size(data_w_t);
notations=cell(z,1);
Boards=cell(z,1);
map2=containers.Map(valueset,keyset);
for i=1:z
    Board=zeros(8,8);
    %get NN responses from data
    input=data_w_t(:,:,i)';
    y_NN1=NN1(input(1:s1-13,:));
    yr_NN1=round(y_NN1)';
    %create Board from NN responses
    for j=1:length(yr_NN1)
        if sum(yr_NN1(j,:))>1
            yr_NN1(j,:)=[1 0 0 0 0 0 0 0 0 0 0 0 0];
        end
        for k=1:13
            if yr_NN1(j,k)==1
                Board(j)=k-1;
            end
        end
    end
    Boards{i,1}=num2cell(Board);
    %recreate labels from boards
    notation='';
    for l=1:8
        zero_cnt=0;
        for m=1:8
            if Board(l,m)==0
                zero_cnt=zero_cnt+1;
                if m==8 && zero_cnt~=0
                    notation=notation+string(zero_cnt);
                end
            else
                if zero_cnt==0
                    notation=notation+map2(Board(l,m));
                else
                notation=notation+string(zero_cnt)+map2(Board(l,m));
                zero_cnt=0;
                end
            end
        end
        zero_cnt=0;
        if l~=8
            notation=notation+'-';
        end
    end
    notations{i}=notation;
end
%%
lab_corr=0;
for i = 1:z
    if notations{i}==labels{i}
        lab_corr=lab_corr + 1;
    end
end
perc_lab_correct=lab_corr/z*100;
%%
%print performance metrics
fprintf('The percentage of square classification was: %0.3f\n',per_correct)
fprintf('The percentage of labels correctly predicted was: %0.3f\n',perc_lab_correct)
