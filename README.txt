Dataset is not bundled in this initial project archive because the runtime used to create it had no internet access.

Run from the project root:
    python download_dataset.py

This fetches the processed Cleveland Heart Disease dataset and creates dataset/heart.csv.
Source dataset: UCI Machine Learning Repository, Heart Disease (Cleveland subset), DOI https://doi.org/10.24432/C52P4X
The download script uses a public GitHub mirror of the UCI processed file.

For a Kaggle submission workflow, you may also download the Kaggle UCI Heart Disease CSV and place it here as heart.csv.
The training code expects the 13 feature columns and a target/num/condition/diagnosis column. See README.md for details.
