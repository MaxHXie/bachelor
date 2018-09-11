import os.path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.manifold import TSNE
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def get_longest_values_and_strings(df, dropna=False):
    '''
    Get the longest string and its lengths of each column in a dataframe

    Arguments:
        df        -- A dataframe containing all data
        dropna    -- A boolean deciding whether or not to drop the empty rows

    Returns:
        longest_values_dict     -- A dictionary containing the length of the longest
                                   value in each column of the dataframe
        longest_strings_dict    -- A dictionary containing the longest string from each
                                   column in the dataframe
    '''
    longest_strings_dict = {}
    longest_values_dict = {}

    i = 0
    for col in df.columns:
        df_temp = df[[col]]
        df_temp = df_temp.astype('str')
        if dropna: df_temp = df_temp.dropna()
        longest_strings_dict[col] = df_temp.iat[df_temp.iloc[:, 0].str.len().idxmax(), 0]
        longest_values_dict[col] = len(longest_strings_dict[col])
        i+=1

    return longest_values_dict, longest_strings_dict



def remove_long_values_anomalies(df):
    '''
    Removes and returns anomaliously long values in columns.

    Arguments:
        df    -- A dataframe containing all data

    Returns:
        df                    -- New dataframe with the longest values removed/ set to NaN
        long_val_anomalies    -- Dictionary as:  column -> [row, anomaly value]

    Loops through and remove any value that is alone in its longest length.
    Loop until there are two or more longest value with same length
    '''
    long_val_anomalies = defaultdict(list)

    for col in df.columns:
        df_temp = df[col].apply(str).map(len)
        df_temp = df_temp.to_frame(name=col)

        while True:
            rowmax = df_temp.max(axis=0)
            if rowmax.any() == 0:
                break
            longest_val_index_list = np.where(df_temp.values == rowmax[:, None])[0]
            if len(longest_val_index_list) <= 15:
                # They are unique if their corresponding list only has one element.
                # Set to NaN in dataframe and add location and value to the outlier dicitonary
                for row in longest_val_index_list:
                    long_anomaly_string = df.at[row, col]
                    long_val_anomalies[col].append([row, long_anomaly_string])
                    df_temp.loc[row, col] = 0
                    df.loc[row, col] = np.NaN
            else:
                break

    return df, long_val_anomalies



def replace_nan_values(df):
    '''
    Arguments:
        df    -- A dataframe containing all data

    Returns:
        unique_symbols_counts_dict    -- A dictionary containing the count of unique
                                         symbols of each column
    '''
    no_nan_df = df.replace(to_replace=np.NaN, value="nan")
    return no_nan_df


def get_unique_symbols_counts_dict(df):
    '''
    Count how many unique symbols there are in total in each column

    Arguments:
        df    -- A dataframe containing all data

    Returns:
        unique_symbols_counts_dict    -- A dictionary containing the count of unique
                                         symbols of each column
    '''
    unique_symbols_counts = {}

    df_temp = df.astype('str')
    unique_symbols_counts_dict = {c : len(set(''.join(df_temp[c]))) for c in df_temp.columns}

    return unique_symbols_counts_dict



def get_number_of_input_units_dict(df, longest_values=[], unique_symbols_counts=[]):
    '''
    Calculate the number of input units each column needs

    Arguments:
        df                       -- A dataframe containing all data
        longest_values           -- A list containing the length of the longest
                                    string in each column
        unique_symbols_counts    -- A list containing the count of unique symbols
                                    in each column
    Returns:
        number_of_input_units_dict    -- A dictionary mapping a cloumn name to amount
                                         of input units that column needs
    '''
    # If longest_values and unique_symbols_counts already have been calculated
    # there is no need to calculate them again.
    number_of_input_units_dict = {}
    if longest_values == [] or unique_symbols_counts == []:
        longest_values_dict, longest_strings_dict = get_longest_values_and_strings(df)
        unique_symbols_counts_dict = get_unique_symbols_counts_dict(df)

    i = 0
    for col in df.columns:
        number_of_input_units_dict[col] = longest_values_dict[col]*unique_symbols_counts_dict[col]
        i+=1

    return number_of_input_units_dict



def create_symbol_maps(one_col_dataframe, column):
    '''
    Map unique symbols to a index and the vice versa. (index for one-hot matrix)

    Arguments:
        one_col_dataframe    -- A dataframe consisting of a single column
        column               -- The name of that column

    Returns:
        index_to_symbol_map    -- A dictionary mapping an index to a unique symbol
        symbol_to_index_map    -- A dictionary mapping a unique symbol to an index
    '''
    df_temp = one_col_dataframe.astype('str')
    unique_symbols_set = set(''.join(df_temp[column]))
    index_to_symbol_map = dict(enumerate(unique_symbols_set))
    symbol_to_index_map = {v: k for k, v in index_to_symbol_map.items()}

    return index_to_symbol_map, symbol_to_index_map



def create_one_hot_matrix(one_col_dataframe, column, symbol_to_index_map, number_of_input_units, output_progress=False):
    '''
    Create a one hot representation of the data in a single column

    Arguments:
        one_col_dataframe        -- A dataframe consisting of a single column
        column                   -- The name of that column
        symbols_to_index_map     -- A dictionary mapping a unique symbol to an index
        number_of_input_units    -- A list containing how many input units each
                                    column needs
        output_progress          -- A boolean deciding whether or not to print out
                                    how many percent is done

    Returns:
        one_hot_matrix    -- A one hot matrix containing all rows in the dataframe
                             in a one hot representation
    '''
    df_temp = one_col_dataframe.astype('str')
    unique_symbols_count = len(symbol_to_index_map)

    #Initialize one_hot_martix
    #Developers comment: This way of initializing the one_hot_matrix is maybe more computationally expensive that it needs to be. Although, this approach makes it really nice to later put in the values by being able to reference them like coordinates.
    one_hot_matrix = []
    new_row = []
    for i in range(df_temp[column].size):
        for j in range(number_of_input_units):
            new_row.append(0)
        one_hot_matrix.append(new_row)
        new_row = []

    percent = 0
    i = 0
    for index, row in df_temp.iterrows():
        j = 0
        for symbol in row[column]:
            col = (j*unique_symbols_count)+symbol_to_index_map[symbol]
            row = i
            one_hot_matrix[row][col] = 1
            j += 1

        if i % (int(df_temp.size/10)+1) == 0 and output_progress == True:
            percent += 10 + int(100/df_temp.size)
            print("[CREATING ONE-HOT][" + column + "] You are now at: " + str(percent) + " %.")

        i += 1

    return one_hot_matrix



def reverse_one_hot_matrix(one_hot_matrix, index_to_symbol_map):
    '''
    Convert one hot matrix to original data

    Arguments:
        one_hot_matrix         -- A one hot matrix containing all rows in the dataframe
                                  in a one hot representation
        index_to_symbol_map    -- A dictionary mapping an index to a unique symbol

    Returns:
        values_array    -- A list containing all data points.
                           (just like each row in a dataframe)

    Arguments1: one_hot_matrix = [
                                 [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0],
                                 [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0],
                                 [0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
                                 [0,0,1,0,0,0,0,0,1,0,0,0,0,0,1],
                                 [0,0,1,0,0,0,0,0,1,0,0,0,1,0,0],
                                 [0,0,0,0,1,0,0,0,1,0,0,0,0,0,1],
                                 [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                                 [1,0,0,0,0,0,0,0,1,0,0,0,0,1,0],
                                 [0,0,0,0,1,0,1,0,0,0,0,0,0,1,0],
                                 ]
    Arguments2: symbol_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}
    Returns1: ['abc','aba','bcd','cde','cdc','ede','abe','add','ebd']
    '''

    values_array = []
    for i in range(len(one_hot_matrix)):
        one_hot_row = one_hot_matrix[i]
        row_value = reverse_one_hot_row(one_hot_row, index_to_symbol_map)
        values_array.append(row_value)

    return values_array



def reverse_one_hot_row(one_hot_row, index_to_symbol_map):
    '''
    Convert one hot matrix to original data

    Arguments:
        one_hot_matrix         -- A one hot matrix containing all rows in the dataframe
                                  in a one hot representation
        index_to_symbol_map    -- A dictionary mapping an index to a unique symbol

    Returns:
        value    -- A string containing all data points.
                    (just like each row in a dataframe)
    '''

    unique_symbols_count = len(index_to_symbol_map)
    value = ""
    value_length = int(len(one_hot_row)/unique_symbols_count)
    for j in range(value_length):
        for k in range(unique_symbols_count):
            if one_hot_row[(j*unique_symbols_count)+k] == 1:
                value += index_to_symbol_map[k]

    return value

def map_contents_to_row_number(df, column):
	'''
    Map the content to its row number (index)

	Arguments:
		df        -- A one-column dataframe to map contents from
		column    -- The name of that one column

	Returns:
		contents_to_row_number_dict    -- A dictionary mapping contents to its row number
	'''

	contents_to_row_number_dict = defaultdict(list)

	for index, row in df.iterrows():
		contents_to_row_number_dict[row[column]].append(index)

	return contents_to_row_number_dict



def get_compressed_one_hot_matrix(one_hot_matrix):
    '''
    Compress the regular format of one_hot_matrix so that it is less expensive to store. Non-lossy compression.

    Arguments:
        one_hot_matrix -- A one hot matrix containing all rows in the dataframe
                          in a one hot representation
    Returns:
        compressed_one_hot_matrix    -- A compressed version of the inputted one_hot_matrix
    '''
    compressed_one_hot_matrix = []
    row_length = len(one_hot_matrix[0])
    compressed_one_hot_matrix.append([row_length])

    for row in one_hot_matrix:
        indices = np.nonzero(row)[0].tolist()
        compressed_one_hot_matrix.append(indices)

    return compressed_one_hot_matrix

def get_decompressed_one_hot_matrix(npz_file_object):
    '''
    Revert the compression algorithm and get the regular data

    Arguments:
        npz_file_object    -- A compressed zip file containing the one-hot matrix

    Returns:
        one_hot_matrix -- A one hot matrix containing all rows in the dataframe
                          in a one hot representation
    '''
    try:
        compressed_one_hot_matrix = npz_file_object.items()[0][1].tolist()
    except FileNotFoundError as e:
        print("[FAILED TO DECOMPRESS] one_hot_matrix")
        print("[ABORTING]")
        return

    if len(compressed_one_hot_matrix[0]) != 1:
        print("[FAILED TO DECOMPRESS] one_hot_matrix")
        print("[ABORTING]")
        return

    one_hot_matrix = []

    row_length = compressed_one_hot_matrix[0][0]
    for i in range(1, len(compressed_one_hot_matrix)):
        row = [0]*row_length
        for index in compressed_one_hot_matrix[i]:
            row[index] = 1
        one_hot_matrix.append(row)

    return one_hot_matrix

def store_array_to_file(file_name, data_array, column="GENERAL"):
    '''
    Takes an array and stores it under the assigned file path/name

    Arguments:
        file_name     -- The string file path and name where the data should be stored
        data_array    -- The array containing the data
    '''
    file_name += '.txt'
    try:
        print("[STORING TO FILE][" + column + "] " + str(file_name))
        np.savetxt(file_name, data_array, fmt='%d')
    except OSError as e:
        print("[FAILED TO STORE][" + column + "] " + str(file_name))



def store_array_to_file_compressed(file_name, data_array, column="GENERAL"):
    '''
    Takes an array and stores it under the assigned file path/name, with a compressed .npz file extension

    Arguments:
        file_name     -- The string file path and name where the data should be stored
        data_array    -- The array containing the data
    '''
    file_name += '.npz'
    try:
        print("[STORING TO FILE][" + column + "] " + str(file_name))
        np.savez_compressed(file_name, data_array)
    except OSError as e:
        print("[FAILED TO STORE][" + column + "] " + str(file_name))



def store_dict_to_file(file_name, dictionary, column="GENERAL"):
    '''
    Takes a dictionary and stores it under the assigned file path/name

    Arguments:
        file_name     -- The string file path and name where the data should be stored
        data_array    -- The array containing the data
    '''
    file_name += '.npy'
    try:
        print("[STORING TO FILE][" + column + "] " + str(file_name))
        np.save(file_name, dictionary)
    except OSError as e:
        print("[FAILED TO STORE][" + column + "] " + str(file_name))

def DATA_PROCESSING(
         file_name='',
         separator=',',
         columns=[],
         compress_one_hot_matrix=True,
         compute_all_columns=False,
         compute_number_of_input_units=True,
         compute_symbol_maps=True,
         compute_contents_to_row_number_map = True,
         compute_one_hot_matrix=True):

    '''
    Handle all data processing, offering much control over what to compute.

    Arguments:
        file_name                             -- Path to file containing data of interest
        separator                             -- A symbol indicating what symbol is separating the columns
        columns                               -- A list specifying which columns to compute on. [OVERWRITES compute_all_columns]
        compute_all_columns                   -- if True: Compute all columns, otherwise fetch from file
        compute_number_of_input_units         -- if True: Compute number of input units required, otherwise fetch from file
        compute_symbol_maps                   -- if True: Compute symbol maps, otherwise fetch from file
        compute_contents_to_row_number_map    -- if True: Compute content to row number map, otherwise fetch from file
        compute_one_hot_matrix                -- if True: Compute one_hot_matrix, otherwise fetch from file

    Returns:
        df.columns                    -- A list of the columns that the algorithm has deicided to compute on

    Store to file:
        number_of_input_units.npy                  -- A dictionary of required number of input units for each column
        index_to_symbol_map_[COLUMN].npy           -- A dictionary mapping one hot encoding index to a symbol in the COLUMN
        symbol_to_index_map_[COLUMN].npy           -- A dictionary mapping a symbol to one hot encoding index in the COLUMN
        contents_to_row_number_map_[COLUMN].npy    -- A dictionary mapping string values to a list of ints, pointing to
                                                      row numbers in the dataset where these values can be found, in the COLUMN
        one_hot_matrix_[COLUMN].txt                -- A one hot matrix containing all rows in the COLUMN dataframe
                                                      in a one hot representation

    '''

    df = pd.read_csv(file_name, encoding='latin-1', dtype=object, sep=separator)

    # Determine which columns to compute on
    if len(columns) != 0:
        df = df[columns]
        if compute_all_columns:
            compute_all_columns = False
            print("Ambigous command: Can't compute on some columns & all columns.")
            print("[COMPUTING COLUMNS] ", end="")
            for column in columns:
                print(str(column), end=", ")

    if len(columns) == 0 and compute_all_columns == False:
        print("Missing command: You have not chosen any columns to compute on.")
        print("[ABORTING]")
        return

    #Replace nan values
    #df = replace_nan_values(df)

    print("[REMOVING LONG VALUES]")
    file_name = "long_values_anomalies"
    extention1 = '.npy'
    extention2 = '.csv'
    df, long_values_anomalies = remove_long_values_anomalies(df)
    store_dict_to_file(file_name, long_values_anomalies)
    df_temp = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in long_values_anomalies.items() ]))
    df_temp.to_csv(file_name + extention2, sep=';')

    ### Get list with number of input units required per column ###
    file_name = "data_parameters/number_of_input_units"
    extention = '.npy'
    if compute_number_of_input_units == False:
        print("[CHECKING] " + file_name + extention)
        if not os.path.isfile(file_name):
            print("[FAILED TO PREPARE] " + file_name + extention)
            print("[RECOMPUTING] number_of_input_units")
            number_of_input_units_dict = get_number_of_input_units_dict(df)
            store_dict_to_file(file_name, number_of_input_units_dict)
        else:
            number_of_input_units_dict = np.load(file_name + '.npy').item()
    else:
        print("[COMPUTING] number_of_input_units_dict")
        number_of_input_units_dict = get_number_of_input_units_dict(df)
        store_dict_to_file(file_name, number_of_input_units_dict)



    ### SEPARATOR ###
    for column in df.columns:
        one_col_dataframe = df[[column]]

        ### Create symbol maps ###
        file_name1 = 'data_parameters/index_to_symbol_map_' + column
        file_name2 = 'data_parameters/symbol_to_index_map_' + column
        extention = '.npy'
        if compute_symbol_maps == False:
            print("[CHECKING][" + column + "] " + file_name1 + extention)
            print("[CHECKING][" + column + "] " + file_name2 + extention)
            if not (os.path.isfile(file_name1 + extention) and os.path.isfile(file_name2 + extention)):
                print("[FAILED TO PREPARE][" + column + "] " + file_name1 + extention)
                print("[FAILED TO PREPARE][" + column + "] " + file_name2 + extention)
                print("[RECOMPUTING][" + column + "] index_to_symbol_map")
                print("[RECOMPUTING][" + column + "] symbol_to_index_map")
                index_to_symbol_map, symbol_to_index_map = create_symbol_maps(one_col_dataframe, column)
                store_dict_to_file(file_name1, index_to_symbol_map, column)
                store_dict_to_file(file_name2, symbol_to_index_map, column)
            else:
                index_to_symbol_map = np.load(file_name1 + extention).item()
                symbol_to_index_map = np.load(file_name2 + extention).item()
        else:
            print("[COMPUTING][" + column + "] index_to_symbol_map, symbol_to_index_map")
            index_to_symbol_map, symbol_to_index_map = create_symbol_maps(one_col_dataframe, column)
            store_dict_to_file(file_name1, index_to_symbol_map, column)
            store_dict_to_file(file_name2, symbol_to_index_map, column)

        ### Compute dictionary that maps string value to its row number in a column ###
        file_name = 'data_parameters/contents_to_row_number_map_' + column
        extention = '.npy'
        if compute_contents_to_row_number_map == False:
            print("[CHECKING][" + column + "] " + file_name + extention)
            if not os.path.isfile(file_name + '.npy'):
                print("[FAILED TO PREPARE][" + column + "] " + str(file_name))
                print("[RECOMPUTING][" + column + "] contents_to_row_number_map")
                contents_to_row_number_map = map_contents_to_row_number(one_col_dataframe, column)
                store_dict_to_file(file_name, contents_to_row_number_map, column)
        else:
            print("[COMPUTING][" + column + "] contents_to_row_number_map")
            contents_to_row_number_map = map_contents_to_row_number(one_col_dataframe, column)
            store_dict_to_file(file_name, contents_to_row_number_map, column)

        # Compute one hot matrix for each column
        file_name = 'data_parameters/one_hot_matrix_' + column
        if compress_one_hot_matrix:
            extention = '.npz'
        else:
            extention = '.txt'
        number_of_input_units = number_of_input_units_dict[column]
        if compute_one_hot_matrix == False:
            print("[CHECKING][" + column + "] " + file_name + extention)
            if not (os.path.isfile(file_name + extention)):
                print("[FAILED TO PREPARE][" + column + "] " + file_name + extention)
                print("[RECOMPUTING][" + column + "] one_hot_matrix")
                one_hot_matrix = create_one_hot_matrix(one_col_dataframe, column, symbol_to_index_map, number_of_input_units, output_progress=True)
                if compress_one_hot_matrix:
                    print("[COMPRESSING][" + column + "] one_hot_matrix")
                    compressed_one_hot_matrix = get_compressed_one_hot_matrix(one_hot_matrix)
                    store_array_to_file_compressed(file_name, compressed_one_hot_matrix, column)
                else:
                    store_array_to_file(file_name, one_hot_matrix, column)
        else:
            print("[COMPUTING][" + column + "] one_hot_matrix")
            one_hot_matrix = create_one_hot_matrix(one_col_dataframe, column, symbol_to_index_map, number_of_input_units, output_progress=True)
            if compress_one_hot_matrix:
                print("[COMPRESSING][" + column + "] one_hot_matrix")
                compressed_one_hot_matrix = get_compressed_one_hot_matrix(one_hot_matrix)
                store_array_to_file_compressed(file_name, compressed_one_hot_matrix, column)
            else:
                store_array_to_file(file_name, one_hot_matrix, column)

    return df.columns

def get_results_dataframe(input_array, reconstruction_array, index_to_symbol_map, contents_to_row_number_map, results_df_rows=100):
    '''
    Taking two arrays, create a dataframe sorting the datapoints with highest reconstruction error, in descending order

    Arguments:
        input_array             -- 2d list containing all real data
        reconstruction_array    -- 2d list containing all reconstructed data
        results_df_rows         -- An int saying how many rows results_df should consist of

    Returns:
        results_df              -- A dataframe sorted by rows with highest error, in descending order
                                   Contains columns 'error' & 'content'
    '''

    mse = np.mean(np.power(input_array - reconstruction_array, 2), axis=1)
    if results_df_rows != 0:
        indices_biggest_values = mse.argsort()[-results_df_rows:][::-1]
    else:
        indices_biggest_values = mse.argsort()[:][::-1]

    mse_array = []
    content_array = []
    content_dictionary = defaultdict(int)
    content_row_numbers = []

    for index in indices_biggest_values:
        one_hot_row = input_array[index]
        content = reverse_one_hot_row(one_hot_row, index_to_symbol_map)
        if content not in content_dictionary:
            mse_array.append(mse[index])
            content_row_numbers.append(contents_to_row_number_map[content])
            content_array.append(content)
            content_dictionary[content] = 1

    results_df = pd.DataFrame({'error': mse_array, 'content': content_array, 'row_numbers':content_row_numbers})

    # Mean error per symbol: error / content_length
    results_df['error'] = results_df['error'].divide(results_df['content'].str.len())
    # Normalize error: error - mean_error
    results_df['error'] = results_df['error'].subtract(results_df['error'].mean(), axis=0)
    results_df = results_df.sort_values(by='error', ascending=False)
    results_df = results_df.reset_index(drop=True)
    return results_df



def output_anomalies(results_df, long_values_anomalies, column=None, threshold_amount=0, threshold_percent=0):
    '''
    Print the examples that are most likely to be anomalies, as defined by the user with a threshold

    Arguments:
        results_df                    -- A sorted dataframe,
                                         starting with the rows with highest reconstruction error.
                                         It contains 2 columns: "Error" and "Content"
        long_val_anomalies            -- Dictionary as:  column -> [row, anomaly value]
		contents_to_row_number_map    -- A dictionary that maps some string content to an integer (its row number)
        column                        -- The string name of the column
		threshold_amount              -- An int telling how many examples should be classed as anomalies
        threshold_percent             -- A float telling how many percent of the dataframe should be classed as anomalies
    '''

    try:
        int(threshold_amount)
        int(threshold_percent)
    except:
        print("You must enter numbers as threshold values.")
        return

    if threshold_amount < 0 and threshold_percent < 0:
        print("Both threshold_amount and threshold_percent is below 0")
        print("[COMPUTING] using threshold_percent = 10")
        threshold_percent = 10
    elif threshold_amount > 0 and threshold_percent > 0:
        print("Both threshold_amount and threshold_percent is greater than 0")
        print("[COMPUTING] using threshold_percent = " + threshold_percent)
        threshold_amount = 0
    elif threshold_amount == 0 and threshold_percent == 0:
        print("Both threshold_amount and threshold_percent is 0")
        print("[COMPUTING] using threshold_percent = 10")
        threshold_percent = 10

    if threshold_percent > 0:
        threshold_amount = int(results_df.size * threshold_percent / 100)

    #Output long values anomalies

    print("[OUTPUTTING] long_values_anomalies")
    for anomaly in long_values_anomalies[column]:
        print(anomaly)

    i = 0
    outputted_content = []
    for index, row in results_df.iterrows():
        content = row['content']
        error = row['error']
        content_row_numbers = row['row_numbers']
        if content not in outputted_content:
            outputted_content.append(content)
            print("Error: ", end="")
            print(str("%.10f" % error) + ", ", end="")
            print("Value: " + str(content))

        i += 1
        if i > threshold_amount:
            break



def plot_difference(input_array, reconstruction_array):
	'''
    Graphically represent the difference between the input_array and the reconstruction_array

	Arguments:
		input_array             -- 2d Array containing all real data
		reconstruction_array    -- 2d Array containing all reconstructed data

	'''

	X = input_array - reconstruction_array

	X_embedded = TSNE(n_components=2).fit_transform(X)
	x = X_embedded[:,0]
	y = X_embedded[:,1]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x,y)
	plt.show()



def represent_one_hot_matrix(one_hot_matrix):
    X = np.array(one_hot_matrix)
    X_embedded = TSNE(n_components=3).fit_transform(X)
    x = X_embedded[:,0]
    y = X_embedded[:,1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    plt.show()
