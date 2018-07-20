clear all;
close all;
clc;

Train_imgFile = 'trainingset\train-images.idx3-ubyte';
Train_labelFile = 'trainingset\train-labels.idx1-ubyte';

Test_imgFile = 'testset\t10k-images.idx3-ubyte';
Test_labelFile = 'testset\t10k-labels.idx1-ubyte';

%//Directly load data in .mat format instead of calling readMNIST
%[train_images train_labels] = readMNIST(Train_imgFile, Train_labelFile, 60000, 0);
%[test_images test_labels] = readMNIST(Test_imgFile, Test_labelFile, 10000, 0);
load('ReadData.mat');

%problem parameters
C = 10;
input_len = 784;
train_len = 60000;
test_len = 10000;

%hyper parameters
step_size = 1E-5;
iter_max = 500;

%model parameters //Two layer multilayer perceptron using sigmoid loss with softmax loss at output.
H = 50; %//Hidden layer node count

%trainable parameters
weights_l1 = randn(H,input_len);
weights_l2 = randn(C,H);

%preallocate memory for performance
arr_train_err = zeros(iter_max,1);
arr_test_err = zeros(iter_max,1);

fc1_output = zeros(H, train_len);
sigmoid_output = zeros(H,train_len);
fc2_output = zeros(C,train_len);
softmax_output = zeros(C,train_len);

grad_weights_l1_vec = zeros(H,train_len);
grad_weights_l1 = zeros(size(weights_l1));
grad_weights_l2 = zeros(size(weights_l2));

target = zeros(C,train_len);
train_predictions = zeros(train_len,1);
test_predictions = zeros(test_len,1);

iter = 0;
while (iter < iter_max)
    iter = iter + 1
    
    %/Training set network execution and gradient calculations/
    fc1_output = weights_l1 * train_images';
        
    for n=1:train_len      
        for j = 1:H
            sigmoid_output(j,n) = 1 / (1 + exp(-fc1_output(j,n)));
        end
    end
    
    fc2_output = weights_l2 * sigmoid_output;
    for n=1:train_len
        for j = 1:C
            softmax_output(j,n) = exp(fc2_output(j,n));
        end 
        softmax_output(:,n) = softmax_output(:,n)/sum(softmax_output(:,n));
        train_predictions(n) = find(softmax_output(:,n)==max(softmax_output(:,n)));
        
        target(train_labels(n)+1,n) = 1; %Convert class label to a indicator based probability value.
        
        for k = 1:C
            for l = 1:C
                grad_weights_l1_vec(k,n) = grad_weights_l1_vec(k,n) + (target(l,n) - softmax_output(l,n)) * weights_l1(l,k);
            end
            grad_weights_l1_vec(k,n) = grad_weights_l1_vec(k,n) * exp(-fc1_output(k,n))/(1+exp(-fc1_output(k,n)))^2;
        end
    end
    
    grad_weights_l1 = -grad_weights_l1_vec * train_images;
    grad_weights_l2 = -(target - softmax_output) * sigmoid_output';
    
    weights_l1 = weights_l1 - step_size * grad_weights_l1;
    weights_l2 = weights_l2 - step_size * grad_weights_l2;
    
    arr_train_err(iter) = length(find(((train_labels + 1) - train_predictions) ~= 0)) / train_len;
    
    
    %/Test set network execution/
    fc1_output = weights_l1*test_images';

     for n=1:test_len      
        for j = 1:H
            sigmoid_output(j,n) = 1/(1 + exp(-fc1_output(j,n)));
        end
    end
    fc2_output = weights_l2 * sigmoid_output;
   
    for n=1:test_len
        for j = 1:C
            softmax_output(j,n) = exp(fc2_output(j,n));
        end 
        softmax_output(:,n) = softmax_output(:,n) / sum(softmax_output(:,n));
        test_predictions(n) = find(softmax_output(:,n) == max(softmax_output(:,n)));
    end
    
    arr_test_err(iter) = length(find(((test_labels+1) - test_predictions)~=0)) / test_len;
    
    disp(['train error: ',num2str(arr_train_err(iter))]);
    disp(['test error: ',num2str(arr_test_err(iter))]);
end


plot(1:iter_max,arr_train_err(1:iter_max)/60000','b',1:iter_max,arr_test_err(1:iter_max)/10000','r');
title 'Plot of Training Errors (Blue) and Test Errors (Red)'
xlabel 'Iteration', ylabel 'Error (Count of Misclassified Images)'
axis([0 iter_max 0 1]);