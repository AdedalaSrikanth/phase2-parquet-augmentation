parquet data augmentation software (phase 3)

overview
this project builds a parquet to parquet data augmentation software. it takes a dataset, converts it into parquet format, performs data augmentation on the minority class, and tracks the source of each augmented sample. the goal is to handle imbalanced data while avoiding data leakage during model evaluation.

dataset
this project uses the stroke dataset. the original csv file is converted into parquet format and used for all processing.

files in this project
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

how the software works

step 1
run build_stroke_input.py to create input.parquet from stroke.csv

step 2
run phase3_minority_augment.py to generate augmented samples for the minority class

step 3
run phase3_model_comparison.py to compare model performance across different methods

step 4
run phase3_minority_model_test.py to evaluate the balanced dataset

augmentation method
this project uses simple gaussian noise to generate new samples. only numeric columns are modified, and the changes are kept small to preserve the original data distribution.

source tracking
each augmented sample stores the source_row_id of the original row it was created from. in group-based mode, all augmented samples from the same source share the same group_id. this ensures correct data splitting during training and testing.

normal vs group-based randomization

normal randomization
augmented samples are treated independently, which can lead to data leakage and overestimated performance

group-based randomization
augmented samples are grouped with their original data, preventing leakage and giving more realistic model performance

results
the original dataset is highly imbalanced, causing poor model performance on the minority class. after applying minority-only augmentation, the dataset becomes balanced and the model performance improves significantly.

limitations
this project does not use smote because it requires combining two samples, which makes source tracking difficult. instead, a single-sample augmentation approach is used.

conclusion
this software demonstrates how data augmentation can be applied in a controlled way to improve model performance while avoiding common issues like data leakage. the use of source tracking and group-based randomization makes the evaluation more reliable.
