# AOD-PM2.5: High-Resolution PM2.5 Mapping Using MAIAC AOD and Machine Learning

This repository contains the codes and framework developed for the paper:  

**Bagheri, H. (2022). A machine learning-based framework for high resolution mapping of PM2.5 in Tehran, Iran, using MAIAC AOD data. *Advances in Space Research, 69*(9), 3333–3349.**  
[https://doi.org/10.1016/j.asr.2022.02.032](https://doi.org/10.1016/j.asr.2022.02.032)

---

## 📖 Overview

This project investigates high-resolution (1 km) daily mapping of ground-level **PM2.5 concentrations** over **Tehran, Iran**, by combining:

- **MAIAC MODIS Aerosol Optical Depth (AOD) retrievals**  
- **Meteorological data (ERA5/ERA5-Land reanalysis)**  
- **Ground-based PM2.5 observations**

A **machine learning-based framework** was developed, consisting of three main stages:

1. **Data preprocessing**  
   - Outlier removal  
   - AOD normalization  
   - Merging Aqua & Terra AODs  
   - Meteorological data interpolation  
   - Feature selection  

2. **Regression modeling**  
   - Tested algorithms: Linear models, SVR, Random Forest, Extra Trees, Deep Learning models, and **XGBoost**.  
   - **XGBoost achieved the best performance** with R² ≈ 0.74 and RMSE ≈ 9 μg/m³.  

3. **Model deployment**  
   - Daily PM2.5 maps at 1 km resolution for Tehran (2013–2019).  
   - Demonstrated capability for different pollution scenarios (clean, moderate, unhealthy).  

---

## 🗂 Data Sources

- **PM2.5 measurements**: Tehran Air Quality Control Company (AQCC)  
- **Satellite AOD (1 km)**: MODIS MAIAC (MCD19A2, Collection 6)  
- **Meteorological data**: ECMWF ERA5 / ERA5-Land (temperature, RH, PBLH, wind, radiation, etc.)  

---

## 🚀 Usage

1. **Preprocess the data**  
   - Apply outlier removal and corrections on PM2.5.  
   - Extract and normalize MAIAC AOD.  
   - Interpolate meteorological variables to station locations.  

2. **Train regression models**  
   - Run provided scripts to compare different models (linear, SVR, RF, XGBoost, etc.).  
   - Evaluate using RMSE, MAE, and R².  

3. **Generate PM2.5 maps**  
   - Deploy the trained model to produce daily PM2.5 grids at 1 km resolution.  
   - Example output maps are included in the manuscript (Fig. 8).  

---

## 📊 Results

- **Best model**: **XGBoost**  
- **Performance on test data**:  
  - RMSE = 8.97 μg/m³  
  - MAE = 6.88 μg/m³  
  - R² = 0.74  
- Successfully produced **daily, 1 km PM2.5 maps** for Tehran (2013–2019), outperforming earlier studies (which used 3–10 km MODIS AOD products).  

---

## 📌 Citation

If you use this code, please cite the following paper:

```bibtex
@article{Bagheri2022,
  title   = {A machine learning-based framework for high resolution mapping of PM2.5 in Tehran, Iran, using MAIAC AOD data},
  author  = {Hossein Bagheri},
  journal = {Advances in Space Research},
  volume  = {69},
  number  = {9},
  pages   = {3333--3349},
  year    = {2022},
  doi     = {10.1016/j.asr.2022.02.032}
}
```

---

## 📬 Contact

For questions or collaboration:  
**Hossein Bagheri**  
Faculty of Civil Engineering and Transportation, University of Isfahan  
📧 h.bagheri.en@gmail.com 
