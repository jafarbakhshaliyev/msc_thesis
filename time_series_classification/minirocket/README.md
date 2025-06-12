# Univariate Time Series Classification with MiniRocket

This subdirectory contains a customized and extended version of the [MiniRocket](https://github.com/angus924/minirocket) model for **univariate time series classification**.

We provide:
- Augmentation-compatible pipeline
- Example shell script for easy reproducibility
- Clear setup instructions for your environment

> 📁 Module path: `time_series_classification/miniRocket/`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jafarbakhshaliyev/msc_thesis.git
cd msc_thesis/time_series_classification/miniRocket/
```

### 2. Install Dependencies

Please download and install the required Python packages manually depending on your setup.

### 3. Download Dataset

Download the UCR Time Series Classification Archive dataset from:

https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

Extract and place the dataset in the following directory structure:

`./minirocket/UCR/`


## ⚙️ Running the Pipeline

We provide a shell script located at `scripts/example.sh` to run MiniRocket with augmentations on a sample dataset.

### ✅ What to Edit

**Open:** `scripts/example.sh`

**Update:**

**Line 9** – set your data directory path:

```bash
cd /your/path/to/data/directory/
```


### Run the Script

```bash
bash scripts/example.sh
```

## 📁 Configure Dataset Path

**Open:** `main.py`

**Update:**

**Line 18** – set your data path to the location where your UCR dataset files are stored:

```python
UCR_PATH  = "/your/full/path/to/data/"
```

Place UCR datasets in the correct subdirectories expected by the code.


## Example Usage

This implementation supports univariate datasets from the UCR Time Series Classification Archive.

**Example datasets tested:**
- Coffee
- ItalyPowerDemand
- TwoLeadECG

Make sure files are correctly named and placed under the expected directories.


## License

This repository is a modified version of [MiniRocket](https://github.com/angus924/minirocket)  
by Angus Dempster, Daniel F. Schmidt, and Geoffrey I. Webb.  
Published in: *ACM SIGKDD 2021 Conference on Knowledge Discovery and Data Mining*, pp. 248–257.

All original and modified code is licensed under the GNU General Public License v3.0 (GPL-3.0).  
Modifications by Jafar Bakhshaliyev (2025) are also released under the same license.

**Third-Party Components:**  
Some augmentation methods are derived from components licensed under Apache License 2.0. See `THIRD_PARTY_LICENSES.txt` for details.

You can find the full license text in the `LICENSE` file.


## 🙏 Acknowledgements

Special thanks to the authors of MiniRocket and the UCR Time Series Classification Archive for making their resources available to the public.

**UCR Dataset Citation:**  
Dau, H. A., Eamonn, K., Kaveh, K., Michael, Y. C.-C., Yan, Z., Shaghayegh, G., Ann, R. C., Yanping, Bing, H., Nurjahan, B., Anthony, B., Abdullah, M., Gustavo, B., and Hexagon-ML (2018). The UCR time series classification archive.

**Time Series Augmentation:**  
Some augmentation methods are adapted from the [time_series_augmentation](https://github.com/uchidalab/time_series_augmentation) repository by Iwana & Uchida.

Iwana, B. K., & Uchida, S. (2021). An empirical survey of data augmentation for time series classification with neural networks. PLOS ONE, 16(7), e0254841. https://doi.org/10.1371/journal.pone.0254841