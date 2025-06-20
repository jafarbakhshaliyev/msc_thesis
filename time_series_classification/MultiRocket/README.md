# Multivariate Time Series Classification with MultiRocket

This subdirectory contains a customized and extended version of the [MultiRocket](https://github.com/ChangWeiTan/MultiRocket) model for **multivariate time series classification**.

We provide:
- Augmentation-compatible pipeline
- Example shell script for easy reproducibility
- Clear setup instructions for your environment

> 📁 Module path: `time_series_classification/MultiRocket/`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jafarbakhshaliyev/msc_thesis.git
cd msc_thesis/time_series_classification/MultiRocket/
```

### 2. Install Dependencies

Install required Python packages using:

```bash
pip install -r requirements.txt
```

If any additional packages are needed, please install them manually depending on your setup.

### 3. Download Dataset

Download the UEA Multivariate Time Series Classification Archive dataset from:

https://www.timeseriesclassification.com/index.php

Extract and place the dataset in the following directory structure:

`./MultiRocket/data/Multivariate_ts/`

## ⚙️ Running the Pipeline

We provide a shell script located at `scripts/example.sh` to run MultiRocket with augmentations on a sample dataset.

### ✅ What to Edit

**Open:** `scripts/example.sh`

**Update:**

**Line 9** – set your data_dir to the location where your .ts files are stored:

```bash
cd /your/path/to/time_series_classification/MultiRocket/
```

**Line 13** – activate your Python virtual environment:

```bash
source /your/path/to/venv/bin/activate
```



### 🏃 Run the Script

```bash
bash scripts/example.sh
```

## 📁 Configure Dataset Path

**Open:** `main.py`

**Update:**

**Line 403** – set your `data_dir` to the location where your `.ts` files are stored:

```python
default = "/your/full/path/to/data/"
```

Place datasets such as `DuckDuckGeese_TRAIN.ts` in the correct subdirectories expected by the code.

## 📂 Folder Structure

```bash
time_series_classification/
└── MultiRocket/
    ├── main.py                 
    ├── multirocket/                
    ├── data/Multivariate_ts/                   
    ├── scripts/
    │   └── example.sh          
    ├── utils/                 
    ├── requirements.txt        
    └── ...
```

##  Example Usage

This implementation supports `.ts` datasets from the UEA Time Series Classification Archive.

**Example datasets tested:**
- DuckDuckGeese
- ERing
- BasicMotions

Make sure files like `*_TRAIN.ts` and `*_TEST.ts` are correctly named and placed under the expected directories.


## License

This repository is a modified version of [MultiRocket](https://github.com/ChangWeiTan/MultiRocket)  
by Chang Wei Tan, Angus Dempster, Christoph Bergmeir, and Geoffrey I. Webb.

It is licensed under the GNU General Public License v3.0 (GPL-3.0).  
All modifications by Jafar Bakhshaliyev (2025) are also released under the same license.

**Third-Party Components:**  
Some augmentation methods in this repository are derived from [time_series_augmentation](https://github.com/uchidalab/time_series_augmentation) which is licensed under Apache License 2.0. See `THIRD_PARTY_LICENSES` for details.

You can find the full license text in the `LICENSE` file.

## 🙏 Acknowledgements

Special thanks to the authors of MultiRocket and the UEA Multivariate Time Series Classification Archive for making their resources available to the public.

UEA Dataset Citation:
Bagnall, A., Dau, H. A., Lines, J., Flynn, M., Large, J., Bostrom, A., Southam, P., and Keogh, E. (2018). The UEA multivariate time series classification archive, 2018.

Time Series Augmentation:
Some augmentation methods are adapted from the [time_series_augmentation](https://github.com/uchidalab/time_series_augmentation) repository by Iwana & Uchida.
Iwana, B. K., & Uchida, S. (2021). An empirical survey of data augmentation for time series classification with neural networks. PLOS ONE, 16(7), e0254841. https://doi.org/10.1371/journal.pone.0254841