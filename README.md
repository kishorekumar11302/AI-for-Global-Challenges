# envi-sustainability-ai

An AI-based environmental intelligence system for sustainability, applied to the Beijing PM2.5 dataset (Liang et al., 2015). The project chains together three components:

1. **Wind-speed forecasting** with a two-layer LSTM, predicting cumulative wind speed (`Iws`) one hour ahead from the preceding 24 hours of weather observations.
2. **Air-pollution forecasting** with a Random Forest, predicting hourly PM2.5 from observed weather, lagged pollution, and the LSTM's wind-speed forecast.
3. **ESG text classification** with the pre-trained zero-shot transformer `facebook/bart-large-mnli`, classifying sustainability text into Environmental, Social, and Governance categories.

The notebook follows a strict chronological evaluation protocol: all train/validation/test splits are time-ordered, scalers are fitted on training data only, and the air model is benchmarked against a persistence baseline with an explicit ablation that isolates the marginal contribution of the wind-speed forecast.

## Repository contents

```
envi-sustainability-ai/
├── Envi_Sustainabilitymodel_Final.ipynb   # main notebook (LSTM, RF, ESG)
├── Group_Report.docx                       # 2,500-word written report
├── data/
│   └── PRSA_data_2010.1.1-2014.12.31.csv   # Beijing PM2.5 dataset
├── requirements.txt                        # Python dependencies
├── .gitignore
└── README.md
```

## Dataset

The Beijing PM2.5 dataset was compiled by Liang et al. (2015) and is hosted by the UCI Machine Learning Repository:

> Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H., & Chen, S. X. (2015). *Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating.* Proceedings of the Royal Society A, 471(2182), 20150257.

The dataset contains 43,824 hourly records collected at the United States Embassy in Beijing between 1 January 2010 and 31 December 2014.

## How to run

Create a Python 3.10+ environment and install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook in Jupyter and run cells top to bottom:

```bash
jupyter notebook Envi_Sustainabilitymodel_Final.ipynb
```

The first run will download the `facebook/bart-large-mnli` model (~1.6 GB) into the local Hugging Face cache. Subsequent runs will use the cached weights.

Total runtime on a modern laptop CPU is approximately 5–10 minutes (LSTM training dominates).

## Reproducibility

Random seeds are fixed (`SEED = 42`) for NumPy, PyTorch, and scikit-learn, and the Hugging Face model revision is pinned (`d7645e1`). Training and test splits are deterministic.

## License

This repository is intended for academic use. The Beijing PM2.5 dataset is redistributed here under the original UCI ML Repository terms; please cite Liang et al. (2015) if you use it.
