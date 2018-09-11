import functions as func
import autoencoders as autoencoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import random

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

def main():
    file_name = 'data/scrambleddata_Allnew2.csv'
    separator = ';'
    columns = ['Col1','Col2','Col3']
    #file_name = 'data/test2.csv'
    #columns = ['col1','col2']
    compress_one_hot_matrix = True
    compute_all_columns = False
    '''
    - If compute_variable is true:     it will compute the variable and store it in a file.
    - If compute_variable is false:    it will try to load the data from a file.
                                       if it fails it will compute the data instead and store it in a file.
    '''
    compute_number_of_input_units = False
    compute_symbol_maps = False
    compute_contents_to_row_number_map = False
    compute_one_hot_matrix = False

    ### START: DATA PROCESS SECTION ###
    columns = func.DATA_PROCESSING(file_name, separator, columns, compress_one_hot_matrix, compute_all_columns, compute_number_of_input_units, compute_symbol_maps, compute_contents_to_row_number_map, compute_one_hot_matrix)
    ### END: DATA PROCESS SECTION ###

    ### START: LOAD DATA SECTION ###
    '''
    Load all data, stored in files:
        number_of_input_units.npy                  -- A list of required number of input units for each column
        index_to_symbol_map_[COLUMN].npy           -- A dictionary mapping one hot encoding index to a symbol in the COLUMN
        symbol_to_index_map_[COLUMN].npy           -- A dictionary mapping a symbol to one hot encoding index in the COLUMN
        contents_to_row_number_map_[COLUMN].npy    -- A dictionary mapping string values to a list of ints, pointing to
                                                      row numbers in the dataset where these values can be found, in the COLUMN
        one_hot_matrix_[COLUMN].txt                -- A one hot matrix containing all rows in the COLUMN dataframe
                                                      in a one hot representation
    '''
    file_path = 'data_parameters/'
    number_of_input_units_dict = np.load(file_path + 'number_of_input_units.npy').item()
    long_values_anomalies = np.load('long_values_anomalies.npy').item()

    for column in columns:
        index_to_symbol_map = np.load(file_path + 'index_to_symbol_map_' + column + '.npy').item()
        symbol_to_index_map = np.load(file_path + 'symbol_to_index_map_' + column + '.npy').item()
        contents_to_row_number_map = np.load(file_path + 'contents_to_row_number_map_' + column + '.npy').item()
        if compress_one_hot_matrix:
            one_hot_matrix = func.get_decompressed_one_hot_matrix(np.load(file_path + 'one_hot_matrix_' + column + '.npz'))
        else:
            one_hot_matrix = np.loadtxt(file_path + 'one_hot_matrix_' + column + '.txt')

        ### END: LOAD DATA SECTION ###

        hidden_layers = 2
        for iteration in range(5):
            ### START: AUTOENCODER SECTION ###
            test_size = 1.0
            while True:
                layer_size_divide_list = np.random.choice([1.3,1.6,1.9], 2)
                if sum(layer_size_divide_list) <= 3.7:
                    break

            learning_rate = np.random.uniform(0.01, 0.02)
            sparsity_target = np.random.uniform(0.10, 0.20)
            sparsity_weight = np.random.uniform(0.40, 0.50)
            l2_reg = 0.0
            n_epochs = 1
            batch_size = 500
            output_progress = False
            output_training_loss = True
            plot_difference = False
            results_df_rows = 0

            print("-----------------ITERATION " + str(iteration) + "-----------------")
            print("Test size: " + str(test_size))
            print("Layer size dividers: ", end="")
            for divider in layer_size_divide_list:
                print(str(divider) + ", ", end="")
            print("")
            print("Learning rate: " + str(learning_rate))
            print("Sparsity target: " + str(sparsity_target))
            print("Sparsity weight: " + str(sparsity_weight))
            print("L2 regularization: " + str(l2_reg))

            '''
            Arguments:
                one_hot_matrix                -- A 2D list, one hot matrix containing all rows in the dataframe
                                                 in a one hot representation
                index_to_symbol_map           -- A dictionary mapping an index to a unique symbol
                contents_to_row_number_map    -- A dictionary that maps some string content to an integer (its row number)
                test_size                     -- A float between 0.0 and 1.0
                layer_size_divide_list        -- A list containing integers of how much each layer shuold be divided with.
                                                 eg. [4,2] with 10 input units will create the network architecture: 10-3-2-3-10
                learning_rate                 -- A float deciding the learning rate of the model
                sparsity_target               -- A float deciding the sparsity target of the model
                sparsity_weight               -- A float deciding the sparsity weight of the model
                l2_reg                        -- A float deciding the l2 regularization of the model
                n_epochs                      -- An int deciding how many epochs to be trained
                batch_size                    -- An int deciding the size of each batch when training
                output_progress               -- A boolean deciding whether or not to output training progress
                output_training_loss          -- A boolean deciding whether or not to output training loss
                plot_difference               -- A boolean deciding whether or not to graphically scatter plot reconstruction error of datapoints
                results_df_rows               -- An int specifying how many rows of errors to output. Set to 0 if you want all rows.
            '''
            results_df = autoencoder.sparse(one_hot_matrix,
                                            index_to_symbol_map,
                                            contents_to_row_number_map,
                                            test_size,
                                            layer_size_divide_list,
                                            learning_rate,
                                            sparsity_target,
                                            sparsity_weight,
                                            l2_reg,
                                            n_epochs,
                                            batch_size,
                                            output_progress,
                                            output_training_loss,
                                            plot_difference,
                                            results_df_rows)
            ### END: AUTOENCODER SECTION ###

            ### START: OUTPUT RESULTS SECTION ###
            print("           -------ANOMALIES-------           ")
            func.output_anomalies(results_df, long_values_anomalies, column=column, threshold_amount=100, threshold_percent=0)
            print()
            ### END: OUTPUT RESULTS SECTION ###

            ### START: SAVE RESULTS SECTION ###
            file_name = "final_results_" + column + ".csv"
            if iteration == 0:
                final_results_df = results_df
                final_results_df['error'] = final_results_df['error'].fillna(0.0)
                final_results_df['error'] = final_results_df['error'].replace('nan', 0.0)
                final_results_df['error'] = final_results_df['error'].replace('NaN', 0.0)
                final_results_df.to_csv(file_name, sep=';')
            else:
                final_results_df_old = pd.read_csv(file_name, sep=';', usecols=['content', 'error', 'row_numbers'], converters={'content' : str})
                final_results_df_old['content'] = final_results_df_old['content'].astype('str')
                df_temp1 = final_results_df_old.sort_values(by='content')
                df_temp2 = results_df.sort_values(by='content')
                df_temp1 = df_temp1.reset_index(drop=True)
                df_temp2 = df_temp2.reset_index(drop=True)
                final_results_df_new = df_temp1.copy()
                df_temp2['error'] = df_temp2['error'].fillna(0.0)
                df_temp2['error'] = df_temp2['error'].replace('nan', 0.0)
                df_temp2['error'] = df_temp2['error'].replace('NaN', 0.0)
                final_results_df_new['error'] = df_temp1['error'].add(df_temp2['error'])
                final_results_df_new = final_results_df_new.sort_values(by='error', ascending=False)
                final_results_df_new = final_results_df_new.reset_index(drop=True)
                final_results_df_new.to_csv(file_name, sep=';')
        ### END: SAVE RESULTS SECTION ###

        ### DONE!!!! ###

main()
