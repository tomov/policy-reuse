import os
import pandas as pd
import numpy as np

# Key columns (bins) == option categories, such that 1) they cover all options and 2) they are mutually exclusive
UNIQUE_CHOICE_COLUMNS = ['gpi zero', 
                          'policy reuse max rew. test', 
                          'policy reuse min rew. test',
                          'policy reuse uncued', 
                          'mb/gpi', 
                          'null trajectories']

def load_data_for_experiment(experiment_version: str = "V0.3_pilot"):
    file_path = os.path.join("data", experiment_version, "summary_subject_x_choice_counts.csv")
    num_options_path = os.path.join("data", experiment_version, "summary_n_valid_trajectories_per_hypothesis.csv")
    return load_data(file_path, num_options_path)

def load_full_data_for_experiment(experiment_version: str = "V0.3_pilot"):
    file_path = os.path.join("data", experiment_version, "summary_subject_x_choice_counts.csv")
    num_options_path = os.path.join("data", experiment_version, "summary_n_valid_trajectories_per_hypothesis.csv")
    return load_data(file_path, num_options_path, choice_columns=None)

def load_full_data_for_experiments(experiment_versions: list[str]):
    all_data = [load_full_data_for_experiment(version) for version in experiment_versions]
    return {
        'df_all': pd.concat([d['df_all'] for d in all_data], ignore_index=True),
        'df_counts': pd.concat([d['df_counts'] for d in all_data], ignore_index=True),
        'df_num_options': pd.concat([d['df_num_options'] for d in all_data], ignore_index=True),
        'choice_columns': all_data[0]['choice_columns'],
        'num_options': all_data[0]['num_options'],
        'counts': np.vstack([d['counts'] for d in all_data])
    }

def load_data(
    file_path="data/summary_subject_x_choice_counts.csv",
    num_options_path="data/summary_n_valid_trajectories_per_hypothesis.csv",
    choice_columns=UNIQUE_CHOICE_COLUMNS
):
    df_all = pd.read_csv(file_path, dtype='int64')
    
    # Create the new column first, then filter
    df_all['policy reuse min rew. test'] = df_all['policy reuse cued'] - df_all['policy reuse max rew. test']
    
    # Load num_options mapping from the trajectories file
    df_num_options_map = pd.read_csv(num_options_path)
    num_options_dict = dict(zip(df_num_options_map['key'], df_num_options_map['value']))
    
    # Get choice columns - use all columns if None, otherwise use specified columns
    if choice_columns is None:
        choice_columns = list(df_all.columns)
    else:
        # Make a copy to avoid modifying the default list
        choice_columns = list(choice_columns)
    
    # Validate that all specified columns exist in the dataframe
    missing_columns = [col for col in choice_columns if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataframe: {missing_columns}")
    
    # Validate that all columns have corresponding num_options entries
    missing_num_options = [col for col in choice_columns if col not in num_options_dict]
    if missing_num_options:
        raise ValueError(f"Missing num_options for columns: {missing_num_options}")
    
    # Get num_options for each column in order
    num_options = [num_options_dict[col] for col in choice_columns]
    
    # Get the empirical choice counts as <n, k>, where n is the number of subjects and k is the number of columns
    df_counts = df_all[choice_columns]
    
    # Get the number of options as <n, k>
    df_num_options = pd.DataFrame(np.tile(num_options, (len(df_counts), 1)), columns=df_counts.columns)
    
    counts = df_counts.to_numpy()

    # Check that all rows sum to the number of test trials
    n_test_trials = int(df_all['random choice'].iloc[0])
    assert all(df_counts[UNIQUE_CHOICE_COLUMNS].sum(axis=1) == n_test_trials), f"Not all rows sum to {n_test_trials}"
    
    return {
        'df_all': df_all,
        'df_counts': df_counts,
        'df_num_options': df_num_options,
        'choice_columns': choice_columns,
        'num_options': num_options,
        'counts': counts
    }

