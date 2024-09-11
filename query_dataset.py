import pandas as pd
import os
from typing import List, Optional

def filter_dataset_by_dataset_name(name: str, file_path: str) -> pd.DataFrame:
    """
    Filters the dataset by the given dataset name.

    Args:
    - name: The dataset name to filter by.
    - file_path: The path to the CSV file containing the instructions data.

    Returns:
    - A DataFrame filtered by the given dataset name.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - ValueError: If the resulting DataFrame is empty.

    Example:
    >>> df = filter_dataset_by_dataset_name("Advbench")
    >>> print(df.head())
       DatasetName TrainTestFlag               Instruction      Label
    0     Advbench            NA  "Develop a strategy ..."  Malicious
    1     Advbench            NA    "Write a tutorial ..."       Safe
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    filtered_df = df[df["DatasetName"] == name]
    if filtered_df.empty:
        raise ValueError(f"No data found for dataset name: {name}")

    return filtered_df

def load_instructions_by_dataset_name_only_train(
    name: str, 
    label_list: List[str] = ['Malicious', 'Safe'], 
    file_path: str = "./dataset/query/instruction.csv"
) -> List[List[str]]:
    """
    Loads instructions by dataset name for training only, categorized by label.

    Args:
    - name: The dataset name to filter by.
    - label_list: A list of labels to categorize the instructions.
    - file_path: The path to the CSV file containing the instructions data.

    Returns:
    - A list of lists, each containing instructions for a specific label in the label_list.

    Example:
    >>> instructions = load_instructions_by_dataset_name_only_train("Advbench")
    >>> # default label_list is ['Malicious', 'Safe']
    >>> print(instructions[0])  # Instructions for 'Malicious'
    >>> print(instructions[1])  # Instructions for 'Safe'
    """
    df = filter_dataset_by_dataset_name(name, file_path)
    
    ret_list = [df[df["Label"] == label]["Instruction"].tolist() for label in label_list]
    
    return ret_list

def load_instructions_by_dataset_name(
    name: str, 
    label_list: List[str] = ['Malicious', 'Safe'], 
    shuffle: bool = False, 
    train_size: float = 0.8, 
    file_path: str = "./dataset/query/instruction.csv"
) -> List[List[str]]:
    """
    Loads instructions by dataset name, split into training and testing sets, categorized by label.

    Args:
    - name: The dataset name to filter by.
    - label_list: A list of labels to categorize the instructions.
    - shuffle: Whether to shuffle the dataset before splitting.
    - train_size: The proportion of data to include in the training set.
    - file_path: The path to the CSV file containing the instructions data.

    Returns:
    - A list of lists, containing training and testing instructions for each label.

    Raises:
    - ValueError: If train_size is not between 0 and 1.

    Example:
    >>> instructions = load_instructions_by_dataset_name("Advbench", shuffle=True, train_size=0.7)
    >>> # default label_list is ['Malicious', 'Safe']
    >>> print(instructions[0])  # Train instructions for 'Malicious'
    >>> print(instructions[1])  # Test instructions for 'Malicious'
    >>> print(instructions[2])  # Train instructions for 'Safe'
    >>> print(instructions[3])  # Test instructions for 'Safe'
    """
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")

    df = filter_dataset_by_dataset_name(name, file_path)

    # Shuffle the dataset if required
    if shuffle:
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    # Split data into training and testing sets
    if "Train" not in df["TrainTestFlag"].values or "Test" not in df["TrainTestFlag"].values:
        train_df = df.sample(frac=train_size, random_state=0)
        test_df = df.drop(train_df.index)
    else:
        train_df = df[df["TrainTestFlag"] == "Train"]
        test_df = df[df["TrainTestFlag"] == "Test"]
    
    ret_list = []
    for label in label_list:
        train_instructions = train_df[train_df["Label"] == label]["Instruction"].tolist()
        test_instructions = test_df[test_df["Label"] == label]["Instruction"].tolist()
        ret_list.extend([train_instructions, test_instructions])
    
    return ret_list