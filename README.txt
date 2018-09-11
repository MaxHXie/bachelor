######### HOW TO RUN THE PROGRAM #########
  - Here are all requirements for running the program. (Also stated below):
    a. Have the files "main.py", "autoencoders.py" and "functions.py" in the same folder.
    b. Have a subfolder named "data_parameters"
    c. Make sure that the data file is correctly referenced in main.py and that it is of the correct format (the first row should contain the name of each column).
    d. Open terminal
    e. Go to the same directory as main.py
    f. type "python main.py"
  - All produced results will be generated in the same directory, as well as the model and other placeholder files.

######### DESCRIPTION OF FILES #########
1. main.py
  - The main program runs on this file.
  - Program Run Settings are tweaked here
    a. file_name: Set this to the location of the data file you want the algorithm to analyze.
    b. separator: Set this to the symbol that is used to separate values in the data file.
    c. columns: Set this to a list containing the names to all columns that you want the algorithm to analyze.
    d. compress_one_hot_matrix: If set to true will compress one_hot_matrix representations. If set true it will also refrain from decompressing all one_hot_matrix files it reads.
    e. compute_all_columns: If set to true will ignore the "column" list and compute the entire data file.
    f. compute_number_of_input_units: If set to true will compute the number of input units needed for each column and store it under the folder "data_parameters". This will be ignored if set to false.
    g. compute_symbol_maps: If set to true will compute the symbol map for each column and store it under the folder "data_parameters" (The symbol map is a dictionary mapping each symbol to its one-hot encoding representation). This will be ignored if set to false.
    h. compute_contents_to_row_number_map: If set to true till compute the contents to row number map and store it under the folder "data_parameters" (The contents to row number map is a dictionary mapping each data point in its original form to its row number in the data file). This will be ignored if set to false.
    i. compute_one_hot_matrix: If set to true will compute all one-hot encoding representations for all data points, combine all data points in a matrix and store it under the folder "data_parameters". This will be ignored if set to false.
  - All compute_ variables can be set to false if they already have been computed. For a given, unchanged, data file, the computed files will always be the same.

  - Model hyperparameters and other settings are tweaked here
    a. hidden_layers: Decides how many hidden layers the network should have. (Recommended is 2)
    b. for iteration in range(5): The number in range() decides number of iterations of retraining which then accumulates on the result. More iterations generate more stable and consistent results on the cost of increased compute time.
    c. test_size: How big share of the data should be part of the result? e.g. 0.8 makes the algorithm randomly choose 80% of all data points and include them in the final results. (It makes no sense to have this number anything else than 1.0)
    d. layer_size_divide_list: Randomly picks 2 float numbers that divides the layers. e.g. if a network has 100 input units and the two randomly picked numbers are 1.3 and 1.9. We get the network architecture: 1000-770-406-770-1000 (1000/1.3 = 770 & 770/1.9 = 406)(if layer_size=2). The quota is always rounded upwards.
    e. learning_rate: Controls how aggresively the model should learn from the data.
    f. sparsity_target: States what degree of sparsity the network should strive for. Sparsity only affects the middle, encoding, layer.
    g. sparsity_weight: Controls how much the model should be penalized when the sparsity_target is missed.
    h. l2_reg: Controls how much l2 regression the model should have (Ridge regression).
    i. n_epochs: Controls how many epochs i.e. how many times the model should train on the entire data set.
    j. batch_size: Controls how big each batch should be for each training step.
    k. output_progress: If set to true will print its training progress directly in terminal.
    l. output_training_loss: If set to true will print the training and sparsity loss for each iteration.
    m. plot_difference: If set to true will plot how big the difference between original data and reconstruction is (This is not functional as of the currect version).
    n. results_df_rows: Controls how many of the top rows of the result should be included. If set to 0, all rows are included.

2. functions.py
  - Contains all miscelleanous funtionality that is needed for the program to run correctly.
  - All functions are commented and their functionality described in the file.

3. autoencoders.py
  - Currently only has the sparse autoencoder as described in the file "sparse". Tensorflow implementation.
  - If architecture is changed, changes needs to be made here by changing the variables: n_hidden1, n_hidden2, n_hidden3 ... weights1, weights2, weights3 ... biases1, biases2, biases3 ... hidden1, hidden2, hidden3 ...
  - Learning parameters are tweaked here
    a. sparsity_loss: Defines how much penalty comes due to failure of hitting the sparsity_target
    b. reconstruction_loss: Defines the objective function
    c. optimizer: Controls which optimizer to use. (tf.train.AdamOptimizer is recommended)
    d. training_op: Controls what to optimize. (optimizer.minimize(loss) is recommended)

######### DESCRIPTION OF DATA FILE #########
  - The data files must have a header row, stating the name of each column.
  - More than that, it doesn't matter how many data points each column consist of or what type.
  - The algorithm might however have some issues with certain symbols.

######### INSTRUCTION FOR DIRECTORY STRUCTURE #########
  - Where the code itself is stored is fully optional.
  - However, the subfolder "data_parameters" is required for the program to function.
  - It is also recommended but not required to store the data in the subfolder "data" (as long as you account for it in the variable "file_name")
