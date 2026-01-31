import pandas as pd
import numpy as np


def load_data(file_path="data/summary_subject_x_choice_counts.csv"):
    df_all = pd.read_csv(file_path, dtype='int64')
    
    # Create the new column first, then filter
    df_all['policy reuse min rew. test'] = df_all['policy reuse cued'] - df_all['policy reuse max rew. test']
    
    # Key columns (bins) == option categories, such that 1) they cover all options and 2) they are mutually exclusive
    choice_columns = ['gpi zero', 
                      'policy reuse max rew. test', 
                      'policy reuse min rew. test',
                      'policy reuse uncued', 
                      'mb/gpi', 
                      'null trajectories']
    # Number of options in each column/bin
    num_options = [1, 1, 1, 1, 1, 4]
    
    # Get the empirical choice counts as <n, k>, where n is the number of subjects and k is the number of columns
    df_counts = df_all[choice_columns]
    assert all(df_counts.sum(axis=1) == 27), "Not all rows sum to 27"
    
    # Get the number of options as <n, k>
    df_num_options = pd.DataFrame(np.tile(num_options, (len(df_counts), 1)), columns=df_counts.columns)
    
    counts = df_counts.to_numpy()
    
    return {
        'df_all': df_all,
        'df_counts': df_counts,
        'df_num_options': df_num_options,
        'choice_columns': choice_columns,
        'num_options': num_options,
        'counts': counts
    }


def load_full_data(
    file_path="data/summary_subject_x_choice_counts.csv",
    num_options_path="data/summary_n_valid_trajectories_per_hypothesis.csv"
):
    df_all = pd.read_csv(file_path, dtype='int64')
    
    # Create the new column first, then filter
    df_all['policy reuse min rew. test'] = df_all['policy reuse cued'] - df_all['policy reuse max rew. test']
    
    # Load num_options mapping from the trajectories file
    df_num_options_map = pd.read_csv(num_options_path)
    # Create a dictionary mapping column names to num_options
    num_options_dict = dict(zip(df_num_options_map['key'], df_num_options_map['value']))
    
    # Get all choice columns (all columns in the dataframe)
    choice_columns = list(df_all.columns)
    
    # Check that all columns have corresponding num_options entries
    missing_columns = [col for col in choice_columns if col not in num_options_dict]
    if missing_columns:
        raise ValueError(f"Missing num_options for columns: {missing_columns}")
    
    # Get num_options for each column in order
    num_options = [num_options_dict[col] for col in choice_columns]
    
    # Get the empirical choice counts as <n, k>, where n is the number of subjects and k is the number of columns
    df_counts = df_all[choice_columns]
    
    # Get the number of options as <n, k>
    df_num_options = pd.DataFrame(np.tile(num_options, (len(df_counts), 1)), columns=df_counts.columns)
    
    counts = df_counts.to_numpy()
    
    return {
        'df_all': df_all,
        'df_counts': df_counts,
        'df_num_options': df_num_options,
        'choice_columns': choice_columns,
        'num_options': num_options,
        'counts': counts
    }

