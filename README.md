<div align="center">

# 🚗 AutoVal
### Used Car Resale Price Estimator

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

**A full-stack web application that predicts the resale price of used cars in real-time.**  
Built with Flask, scikit-learn, and a dark-themed interactive dashboard.

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo Preview](#-live-demo-preview)
- [Features](#-features)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)

---

## 🧠 Overview

**AutoVal** takes 10 real-world vehicle parameters and predicts the resale price using two trained ML models — a **Random Forest Regressor** (R² = 0.938) and a **Linear Regression** baseline — both trained on **15,397 real CarDekho transactions**.

The app features a live metrics dashboard that shows actual vs predicted prices, residual distributions, feature importance, and prediction error by vehicle age — all fetched dynamically from the Flask backend.

---

## 🖥️ Live Demo Preview

> The app runs locally. Below are actual screenshots of the UI:

<br/>

### 🔷 Price Prediction Form

<p align="center">
  <img src="assets/form.png" width="850"/>
  <br/>
  <em>Real-time car price estimation with confidence interval and feature insights</em>
</p>

<br/>

### 📊 Metrics Dashboard

<p align="center">
  <img src="assets/dashboard.png" width="850"/>
  <br/>
  <em>Interactive ML performance dashboard with multiple evaluation metrics and charts</em>
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Real ML Prediction** | Random Forest (R²=0.938) + Linear Regression trained on 15,397 records |
| ⚡ **Instant Estimates** | Sub-second predictions via Flask REST API |
| 📊 **Live Metrics Dashboard** | 5 interactive Chart.js charts — all data fetched from backend |
| 🔍 **Price Influencers** | Shows which factors pushed the price up or down |
| 📉 **Confidence Score** | Per-prediction confidence interval with animated bar |
| 🔀 **Algorithm Toggle** | Switch between RF and LR — charts update live |
| 📱 **Responsive UI** | Works on desktop and mobile |
| 🔁 **Retrain Anytime** | Run `train.py` with new data to update everything |

---

## 📈 Model Performance

Both models trained on CarDekho dataset · 80/20 train-test split · random_state=42

| Metric | 🌲 Random Forest | 📉 Linear Regression |
|:---|:---:|:---:|
| **R² Score** | **0.938** | 0.689 |
| **MAE (₹ Lakhs)** | **0.94** | 2.45 |
| **RMSE (₹ Lakhs)** | **1.98** | 4.46 |
| **MAPE** | **13.4%** | 40.1% |

<br/>

### 🏆 Top Feature Importances (Random Forest)
