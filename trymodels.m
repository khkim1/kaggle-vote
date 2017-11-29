data = csvread('train_2008.csv', 1);
data = data(:, 4:end);

x_train = data(:, 1:end-1);
y_train = data(:, end);