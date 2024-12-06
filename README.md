## **Steps to Run Wiki Extract**

### **1. Clone the Repository**
```bash
git clone https://github.com/s-ankur/wikiextract
cd wikiextract
```

### **2. Set Up the Environment**

Run the `setup.sh` script to set up the necessary dependencies, repositories, and data dumps.

```bash
bash
Copy code
bash setup.sh
```

### **3. Cleanup After Failed Runs**

If any run fails, execute the following commands to clean intermediate outputs:

```bash
bash
Copy code
rm hindi.output hindi-pos-tagger-3.0/hindi.input.txt
rm extracted
```

### **4. Run the Main Processing Script**

To generate the augmented dataset, execute:

```bash
bash
Copy code
bash run_hiwiki.sh
```

This step may take a significant amount of time, depending on your system.

### **5. Sample and Inspect Output**

You can inspect a sample of the generated sentences using:

```bash
bash
Copy code
head -4000 hiwiki.augmented.edits | python scripts/convert_to_wdiff.py | shuf -n 40
```

Check the word count for the generated file:

```bash
bash
Copy code
wc hiwiki.augmented.edits
```

### **6. Split Data into Training and Validation Sets**

Split the dataset into training and validation files:

```bash
bash
Copy code
head -7823271 hiwiki.augmented.edits > train
tail -1564656 hiwiki.augmented.edits > val
mkdir -p data
```

### **7. Shuffle the Data**

Before training, shuffle the datasets:

```bash
bash
Copy code
bash shuffle.sh data/train_merge 42
bash shuffle.sh data/valid 42
```

### **8. Move Processed Data**

Move the shuffled data to the `data` directory:

```bash
bash
Copy code
wc data/*
mv data ../data
```

Wiki Edits 2.0
==============

A collection of scripts for automatic extraction of edited sentences from text
edition histories, such as Wikipedia revisions.

This repository contains a Jupyter Notebook titled wikiedits.ipynb. The notebook analyzes Wikipedia edits, including data exploration and visualization to identify key editing patterns.

Requirements:
1. Python 3.x
2. Jupyter Notebook
3. Python libraries listed in the notebook (e.g., pandas, matplotlib, etc.)

Install the necessary libraries using:
`pip install -r requirements.txt`

Usage:

1. Clone this repository:
`git clone https://github.com/yourusername/wikiedits.git`

2. Navigate to the directory:
`cd wikiedits`

3. Start Jupyter Notebook:
`jupyter notebook wikiedits.ipynb`

4. Open the notebook and execute the cells.