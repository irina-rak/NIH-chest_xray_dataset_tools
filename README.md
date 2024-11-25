# NIH Chest X-ray Dataset

This repository contains a custom tool for the NIH Chest X-ray dataset, which is a large collection of chest radiographs (X-rays) provided by the National Institutes of Health (NIH) Clinical Center and is intended for research purposes. It is one of the largest publicly available chest X-ray datasets and includes images with labels for 14 different thoracic diseases.

This repository is going to be used as a template for creating a Python processing pipeline package for the NIH Chest X-ray dataset. For now, it contains only the source code for the data processing tool. The package will be further developed to include additional functionalities and documentation.

Note that this package is not intended to be a replacement for the original dataset, but rather a tool to facilitate the processing of the data. The package provides a command-line interface for splitting the dataset into train, validation, and test sets, as well as for resizing the images and generating the corresponding CSV files.

```plaintext
Disclaimer:
- This package is intended for educational and research purposes only. Please refer to the original dataset for the most up-to-date information and terms of use.
- This package is not affiliated with the NIH Clinical Center or the original creators of the dataset.
- As the images are about 42GB in size, they are not included in this repository. Please download the dataset from the original source.
```

## Dataset Description

The dataset includes X-ray images along with labels for 14 different thoracic diseases. The images are in PNG format and are organized into 12 folders. The labels are provided in CSV format and contain the following columns: `Image Index`, `Finding Labels`, `Follow-up #`, `Patient ID`, `Patient Age`, `Patient Gender`, `View Position`, `OriginalImageWidth`, `OriginalImageHeight`, `OriginalImagePixelSpacing_x`, `OriginalImagePixelSpacing_y`.

## Structure

The package is organized as follows:

```
NIH_Chest_Xray/
├── configs/
├── dataset/
├── src/
│   ├── commands/
│   ├── data_processor/
│   ├── utils/
│   ├── console.py
│   └── main.py
├── .gitignore
├── README.md
├── pyproject.toml
├── poetry.lock
```
- `configs/`: Contains the .yml configuration files as input to the data processing tool.
- `dataset/`: Contains the split files (train, validation, test), provided in CSV format.
- `src/`: Contains the source code for the dataset processing tool.
    - `commands/`: Contains the command-line interface commands.
    - `data_processor/`: Contains the data processing functions.
    - `utils/`: Contains utility functions.
    - `console.py`: Contains the main console script.
    - `main.py`: Contains the main script for processing the dataset.
- `.gitignore`: Specifies which files and directories to ignore in Git.
- `README.md`: This file.
- `pyproject.toml`: Contains the project metadata and dependencies.
- `poetry.lock`: Contains the exact versions of the dependencies.

## Usage

To use this package, you need to have Python and Poetry installed on your system. You can install Poetry by following the instructions [here](https://python-poetry.org/docs/).

I highly recommend creating a virtual environment before installing the dependencies, to avoid conflicts with other packages. You can create a virtual environment using miniconda ([installation instructions](https://docs.conda.io/en/latest/miniconda.html)) or venv ([installation instructions](https://docs.python.org/3/library/venv.html)).

Once you have set up the virtual environment and installed Poetry, you can install the dependencies by running the following command:

```bash
poetry install
```

To run the data processing tool, you can either use the Python script or the Poetry command. The script takes a configuration file as input, which specifies the paths to the input and output directories, as well as the split ratios for the train, validation, and test sets. You can find an example configuration file in the `configs/` directory.

To run the script using Python, use the following command:

```bash
poetry run python src/main.py --config configs/config.yml
```

Alternatively, you can use the Poetry command to run the script:

```bash
nih_processor_app preprocessing launch configs/config.yml
```

## License

This dataset is licensed under the GNU General Public License v3.0. For more details, please refer to the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.en.html) file.

## Acknowledgements

I would like to thank the NIH Clinical Center for providing this dataset and making it available to the research community. For more information, please visit the [original dataset link](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community).

For more information, please visit the [NIH website](https://www.nih.gov/).

As this package is based on the NIH Chest X-ray database, please cite the following paper if you use this dataset in your research:

```
@article{Wang2017ChestXRay8HC,
    title={ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
    author={Xiaosong Wang and Yifan Peng and Le Lu and Zhiyong Lu and Mohammadhadi Bagheri and Ronald M. Summers},
    journal={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2017},
    pages={3462-3471},
    url={https://api.semanticscholar.org/CorpusID:263796294}
}
```