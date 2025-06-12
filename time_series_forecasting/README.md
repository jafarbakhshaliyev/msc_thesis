# Time Series Forecasting with Data Augmentation

This directory contains implementations of data augmentation techniques for time series forecasting.

We provide:
- Multiple augmentation strategies for improved forecasting performance
- Easy-to-use pipeline with configurable parameters
- Support for various time series forecasting datasets
- Comprehensive experimental setup

> 📁 Module path: `time_series_forecasting/`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jafarbakhshaliyev/msc_thesis.git
cd msc_thesis/time_series_forecasting/
```

### 2. Install Dependencies

Install required Python packages using:

```bash
pip install -r requirements.txt
```

If any additional packages are needed during execution, please install them manually:

```bash
pip install [package_name]
```

### 3. Download Dataset

Download the time series forecasting datasets from:

https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

Extract and place the datasets in the following directory structure:
`./time_series_forecasting/dataset/`


### ⚙️ Usage
Using Example Scripts
We provide pre-configured shell scripts for each model under the scripts/ directory. Each script includes optimized parameters and augmentation settings:

```bash
# Run DLinear model
bash scripts/example_DLinear.sh

# Run LightTS model
bash scripts/example_LightTSr.sh

# Run PatchTST model
bash scripts/example_PatchTST.sh

# Run TiDE model
bash scripts/example_TiDE.sh

# Run TSMixer model
bash scripts/example_TSMixer.sh
```

### Legal and Ethical Disclosure

#### Code Origins
This project implements algorithms from published research papers. Some initial 
code was adapted from repositories that lacked explicit licenses.

#### Our Approach
We have substantial change to the main work and agumentations file.

#### Usage Recommendations
- Academic research and education
- Learning and experimentation  
- Commercial use: Seek legal advice
- Distribution: Include this notice

#### Contact
If you are an original author of the some parts of code and have concerns, 
please contact  jafar.bakhshaliyev@gmail.com.

### 🙏 Acknowledgements

This implementation builds upon and extends the following research works:

**Dominant Shuffle:**
```bibtex
@misc{zhao2024dominantshufflesimplepowerful,
  title={Dominant Shuffle: A Simple Yet Powerful Data Augmentation for Time-series Prediction}, 
  author={Kai Zhao and Zuojie He and Alex Hung and Dan Zeng},
  year={2024},
  eprint={2405.16456},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2405.16456}
}
```

FrAug:
```bibtex
@misc{chen2023fraugfrequencydomainaugmentation,
  title={FrAug: Frequency Domain Augmentation for Time Series Forecasting}, 
  author={Muxi Chen and Zhijian Xu and Ailing Zeng and Qiang Xu},
  year={2023},
  eprint={2302.09292},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2302.09292}
}
```
Special thanks to the authors for their contributions to time series forecasting and data augmentation research. There are other methods we either replicated the augmentation methods by ourselves or some repositories which are mentioned in time_series_classifcation which are modified for forecasting. To see the full reference, please refer to the PDF report in ./latex folder.

Code Implementation:

Original Dominant Shuffle implementation: https://github.com/zuojie2024/dominant-shuffle
Extended and modified by Jafar Bakhshaliyev (2025)


### 📝 Citation
If you use this code in your research, please cite the original papers mentioned above along with this repository.
