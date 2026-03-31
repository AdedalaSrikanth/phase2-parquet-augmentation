# parquet augmentation project

## overview

This project builds a parquet to parquet data augmentation software for tabular datasets.

The goal of this phase is to implement the core logic for generating augmented samples while keeping track of the original data.

---

## files

- build_input.py  
  converts the csv dataset into input.parquet

- phase2_parquet_augment.py  
  main augmentation program

- ai4i2020.csv  
  original dataset file

- input.parquet  
  dataset used as input for augmentation

- augmented_output.parquet  
  final dataset with original and augmented rows

---

## what the software does

The program reads a parquet file from the same folder and creates new synthetic rows by adding small noise to numeric columns.

It also keeps track of where each augmented row came from.

---

## how it works

1. read input.parquet  
2. identify numeric columns  
3. create multiple augmented copies of each row  
4. add small random noise to feature values  
5. keep tracking columns like source_row_id  
6. save the result as augmented_output.parquet  

---

## randomization modes

The software supports two modes:

normal  
- treats all rows independently  
- group information is ignored  

group_based  
- preserves group_id  
- keeps related rows together  

---

## current phase

This phase focuses on:

- building the augmentation software  
- supporting parquet input and output  
- adding row tracking  
- supporting randomization modes  

---

## not implemented yet

The following will be done in phase 3:

- comparing model performance with and without augmentation  
- comparing normal vs group-based randomization  

---

## notes

- input file must be in the same directory  
- output file is saved in the same directory  
- only numeric columns are augmented  
