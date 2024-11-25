from os import scandir
from os.path import isdir, join
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold


def read_csv(file_path: str) -> pd.DataFrame:
    """Read a csv file and return a pandas DataFrame

    Args:
    -----
        file_path (str): Path to the csv file
    """
    return pd.read_csv(file_path)


def get_unique_patients(df: pd.DataFrame) -> List[str]:
    """Get unique patient IDs from a DataFrame

    Args:
    -----
        df (pd.DataFrame): DataFrame containing the data
    """
    return df["Patient ID"].unique()


def stratified_train_test_split(
    df_file: Union[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train, validation, and test sets

    Args:
    -----
        file_path (str): Path to the csv file or the DataFrame containing the data
    """
    if isinstance(df_file, str):
        df = read_csv(df_file)
    else:
        df = df_file.copy()

    # Create 5 splits and merge the 4 splits to create the training set and the last split to create the test set
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_df = None
    val_df = None
    for train_index, test_index in skf.split(
        df, df["Finding Labels"], groups=df["Patient ID"]
    ):
        train_df = df.iloc[train_index]
        val_df = df.iloc[test_index]
        break

    return train_df, val_df


def random_fl_split(
    n_splits: int,
    df: pd.DataFrame,
    unbalanced: bool = False,
    extreme: bool = False,
    target_clients: Union[int, float] = 0,
) -> Tuple[pd.DataFrame]:
    """Split the dataset into n_splits clients using random assignment with optional extreme unbalancing and reassignment of excluded patients.

    Args:
    -----
        n_splits (int): Number of clients to split the data into.
        df (pd.DataFrame): DataFrame containing the data.
        unbalanced (bool): Whether to split the data into unbalanced clients.
        extreme (bool): Whether to apply extreme unbalancing.
        target_clients (Union[int, float]): Number or percentage (if float) of clients
                                             to be subjected to extreme unbalancing.

    Returns:
    --------
        Tuple[pd.DataFrame]: Tuple containing the DataFrames for each client.
    """
    patients = df["Patient ID"].unique()
    np.random.shuffle(patients)

    assert (
        target_clients < n_splits
    ), "Number of target clients cannot exceed the number of splits."
    assert target_clients > 0, "Number of target clients must be greater than 0."
    assert n_splits < len(
        patients
    ), "Number of splits cannot exceed the number of unique patients."

    if unbalanced:
        random_points = np.sort(
            np.random.choice(range(1, len(patients)), n_splits - 1, replace=False)
        )
        split_sizes = np.diff([0] + random_points.tolist() + [len(patients)])
    else:
        base_size = len(patients) // n_splits
        split_sizes = np.array([base_size] * n_splits)
        remainder = len(patients) % n_splits
        if remainder > 0:
            indices = np.random.choice(range(n_splits), size=remainder, replace=False)
            split_sizes[indices] += 1

    split_points = np.cumsum(split_sizes)[:-1]
    clients = np.split(patients, split_points)

    if extreme:
        # Determine the clients to apply extreme unbalancing to
        target_clients_count = target_clients
        if isinstance(target_clients, float):  # If it's a percentage
            target_clients_count = max(1, int(n_splits * target_clients))

        # Special case: Ensure all clients are included if there are only 2 clients
        if n_splits == 2:
            target_clients_count = n_splits

        target_clients_indices = (
            np.random.choice(range(n_splits), size=target_clients_count, replace=False)
            if target_clients_count > 0
            else []
        )

        unique_classes = df["Finding Labels"].str.split("|").explode().unique()
        unique_classes = unique_classes[unique_classes != "No Finding"]

        to_swap = np.array_split(
            np.random.permutation(unique_classes), target_clients_count
        )

        # remaining_patients = {}
        patients_to_swap = {}
        for tc_idx in target_clients_indices:
            # print(f"Excluding classes {to_swap[tc_idx]} from client {tc_idx}")
            client_df = df[df["Patient ID"].isin(clients[tc_idx])]
            # classes_to_exclude = to_swap.pop()
            classes_to_exclude = to_swap[tc_idx]
            excluded_patients = client_df[
                client_df["Finding Labels"].str.contains("|".join(classes_to_exclude))
            ]["Patient ID"].unique()
            # clients[tc_idx] = np.setdiff1d(clients[tc_idx], excluded_patients)
            # remaining_patients = np.setdiff1d(patients, np.concatenate(clients))
            patients_to_swap[tc_idx] = excluded_patients
            clients[tc_idx] = np.setdiff1d(clients[tc_idx], excluded_patients)

        # Shift idx of to_swap to avoid reassigning the same patients
        clients = clients[::-1]
        for idx in target_clients_indices:
            clients[idx] = np.concatenate((clients[idx], patients_to_swap[idx]))

    client_dfs = [
        df[df["Patient ID"].isin(client)].reset_index(drop=True) for client in clients
    ]

    return tuple(client_dfs)


def filter_data(df: pd.DataFrame, keep_single_label: bool) -> pd.DataFrame:
    median_age = np.ceil(df.groupby("Patient ID")["Patient Age"].median()).astype(int)

    outlier_patients = median_age[median_age > 100].index
    for pid in outlier_patients:
        patient_data = df[df["Patient ID"] == pid]
        num_images = patient_data["Image Index"].count()

        if num_images > 1:
            # Replace patient age > 100 with the minimum patient age for those with multiple images
            min_age = patient_data["Patient Age"].min()
            if min_age <= 100:
                df.loc[
                    (df["Patient ID"] == pid) & (df["Patient Age"] > 100), "Patient Age"
                ] = min_age
        else:
            # Drop patients with median age > 100 and only one image
            df.drop(patient_data.index, inplace=True)

    # Now handle remaining outliers (Patient Age still > 100)
    remaining_outliers = df[df["Patient Age"] > 100]
    for pid in remaining_outliers["Patient ID"].unique():
        df.loc[(df["Patient ID"] == pid) & (df["Patient Age"] > 100), "Patient Age"] = (
            median_age.loc[pid]
        )

    # Keep only images with 1 label
    if keep_single_label:
        df = df[df["Finding Labels"].str.contains("\|") == False]
    return df


def get_dummy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the labels into one-hot encoded labels.

    Args:
    -----
        data (pd.DataFrame): Dataframe containing the labels.

    Returns:
    --------
        pd.DataFrame: Dataframe with one-hot encoded labels.
    """
    data_df = df.copy()
    labels = (
        pd.get_dummies(df["Finding Labels"].str.split("|").explode())
        .groupby(level=0)
        .sum()
    )
    data_df = pd.concat([data_df, labels], axis=1)
    return data_df


# --- Plotting functions ---


def plot_age_distribution(
    df,
    filter_outliers=False,
    keep_single_label: bool = False,
    # add_path: bool = False,
    dataset_dir: str = "",
    plot_title: str = "Gender distribution by age",
) -> pd.DataFrame:
    """Plot the age distribution of patients in the dataset.

    Args:
    -----
        df (pd.DataFrame): DataFrame containing the data
        filter_outliers (bool): Whether to filter out outliers (i.e. patients with age > 100)
        keep_single_label (bool): Whether to keep only images with a unique label
        add_path (bool): Whether to add the path of the image to the DataFrame
        dataset_dir (str): Path to the dataset directory
        plot_title (str): Title of the plot
    """
    if filter_outliers:
        filtered_data = filter_data(df.copy(deep=True), keep_single_label)
    else:
        filtered_data = df.copy(deep=True)

    filtered_data.drop(["OriginalImage[Width", "Height]", "OriginalImagePixelSpacing[x', 'y]", "Unnamed: 11"], axis=1, inplace=True, errors="ignore")

    # if add_path:
    if dataset_dir:
        assert isdir(dataset_dir), "Invalid dataset directory."

        ds_dirs = [f.name for f in scandir(dataset_dir) if isdir(f)]

        for _dir in ds_dirs:
            _img_dir = join(dataset_dir, _dir, "images")
            image_files = [
                f.name
                for f in scandir(_img_dir)
                if f.is_file() and f.name.endswith(".png")
            ]
            filtered_data.loc[
                filtered_data["Image Index"].isin(image_files), "Path"
            ] = _img_dir

    # Group by Patient ID and aggregate median
    aggregated_data = filtered_data.groupby("Patient ID").agg(
        {"Patient Age": "median", "Patient Gender": "first"}
    )

    colors = sns.color_palette("deep", 2)
    plt.figure(figsize=(20, 5))
    ax = sns.histplot(
        data=aggregated_data,
        x="Patient Age",
        hue="Patient Gender",
        multiple="dodge",
        kde=True,
        bins=len(aggregated_data["Patient Age"].unique()),
        hue_order=["M", "F"],
        palette=colors,
    )

    # Create custom legend labels with patient counts
    gender_counts = aggregated_data["Patient Gender"].value_counts()
    # custom_labels = [f"{gender}: {count} ({count / len(aggregated_data) * 100:.2f}%)" for gender, count in gender_counts.items()]
    custom_labels = [
        f"{t.get_text()}: {gender_counts[t.get_text()]} ({gender_counts[t.get_text()] / len(aggregated_data) * 100:.2f}%)"
        for t in ax.legend_.texts
    ]

    for t, l in zip(ax.legend_.texts, custom_labels):
        t.set_text(l)

    ax.set_xlim(left=0)
    ax.set(xlabel="Age")
    ax.set(ylabel="Frequency")
    ax.set_title(plot_title)
    plt.xticks(rotation=45)

    return filtered_data


def plot_disease_distribution(data):
    disease = data["Finding Labels"].str.split("|").explode()
    disease = disease.value_counts()

    # colors = sns.color_palette("pastel")[0:len(disease)]
    colors = sns.color_palette("deep", len(disease))

    plt.figure(figsize=(15, 5))
    ax = sns.barplot(
        x=disease.index, y=disease.values, palette=colors, hue=disease.index
    )

    ax.set(ylabel="Frequency")
    ax.set_title("Diseases distribution")
    plt.xticks(rotation=45)

    for i in ax.containers:
        ax.bar_label(i, label_type="edge")
