Parquet data augmentation software (phase 3)

Overview
This project builds a parquet to parquet data augmentation software. It takes a dataset, converts it into parquet format, performs data augmentation on the minority class, and tracks the source of each augmented sample. The goal is to handle imbalanced data while avoiding data leakage during model evaluation.

Dataset
This project uses the stroke dataset. The original csv file is converted into parquet format and used for all processing.

Files in this project
build_stroke_input.py
converts stroke.csv into input.parquet

parquet_augment.py
main augmentation logic for generating new samples

phase3_model_comparison.py
compares model performance using original data, normal augmentation, and group-based augmentation

phase3_minority_augment.py
generates augmented samples only for the minority class to balance the dataset

phase3_minority_model_test.py
tests model performance on the balanced dataset

stroke.csv
original dataset

input.parquet
processed input dataset

augmented_minority_only.parquet
final augmented dataset with balanced classes

model_comparison_results.csv
results comparing different augmentation methods

minority_model_results.csv
results after balancing the dataset

How the software works

Step 1
Run build_stroke_input.py to create input.parquet from stroke.csv

Step 2
Run phase3_minority_augment.py to generate augmented samples for the minority class

Step 3
Run phase3_model_comparison.py to compare model performance across different methods

Step 4
Run phase3_minority_model_test.py to evaluate the balanced dataset

Augmentation method
This project uses simple gaussian noise to generate new samples. Only numeric columns are modified, and the changes are kept small to preserve the original data distribution.

Source tracking
Each augmented sample stores the source_row_id of the original row it was created from. In group-based mode, all augmented samples from the same source share the same group_id. This ensures correct data splitting during training and testing.

Normal vs Group-based randomization

Normal randomization
Augmented samples are treated independently, which can lead to data leakage and overestimated performance

Group-based randomization
Augmented samples are grouped with their original data, preventing leakage and giving more realistic model performance

Results
The original dataset is highly imbalanced, causing poor model performance on the minority class. After applying minority-only augmentation, the dataset becomes balanced and the model performance improves significantly.

Limitations
This project does not use smote because it requires combining two samples, which makes source tracking difficult. Instead, a single-sample augmentation approach is used.

Conclusion
This software demonstrates how data augmentation can be applied in a controlled way to improve model performance while avoiding common issues like data leakage. The use of source tracking and group-based randomization makes the evaluation more reliable.
