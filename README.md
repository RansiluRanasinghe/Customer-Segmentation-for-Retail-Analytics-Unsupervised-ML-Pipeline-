# 🎯 Customer Segmentation Using K-Means

**Industry-Grade Unsupervised Machine Learning Pipeline**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=google-colab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

This project implements an **end-to-end, industry-aligned customer segmentation pipeline** using unsupervised learning (K-Means clustering) on real-world retail transaction data.

The goal is to simulate how machine learning engineers build customer segmentation models in production environments, focusing on:
- Data validation
- Feature engineering
- Model robustness
- Interpretability
- Reproducibility

Rather than optimizing for predictive accuracy, the project emphasizes **business interpretability and operational usefulness**, which is how unsupervised ML systems are evaluated in industry.

> 🚫 Not a classroom demo  
> ✅ A **realistic ML engineering pipeline** built with production standards in mind

---

## 🧠 Business Problem

Retail and e-commerce organizations often need to understand customer behavior, but face challenges such as:

- ❌ **No labeled customer segments**
- ❌ **Large volumes** of transactional data
- ❌ **Highly diverse** purchasing patterns
- ❌ **Changing customer behavior** over time

### Why This Matters

Manual analysis **does not scale**, and predefined customer categories quickly become **outdated**.

### The Need

Organizations require systems that can:
- ✅ Automatically **discover customer behavior patterns**
- ✅ **Group customers** based on real purchasing activity
- ✅ Support **marketing, retention, and strategic decisions**

---

## 💡 Solution Overview

This project addresses the problem by:

1. Transforming raw transaction data into customer-level features
2. Engineering **RFM (Recency, Frequency, Monetary)** metrics
3. Applying **K-Means clustering** to discover natural customer segments
4. Validating and interpreting clusters from both technical and business perspectives

The result is a **production-style customer segmentation model** that mirrors how unsupervised ML is applied in real organizations.

---

## 📊 Dataset

### Source
**Online Retail II Dataset** (UCI Machine Learning Repository, via Kaggle)

### Description
Transactional data from a UK-based online retailer between **2009 and 2011**.

### Key Fields Used

| Field | Description |
|-------|-------------|
| `InvoiceNo` | Unique transaction identifier |
| `Quantity` | Number of items purchased |
| `UnitPrice` | Price per item |
| `InvoiceDate` | Transaction timestamp |
| `CustomerID` | Unique customer identifier |

### Important Note

> **No labels are used for training.** The model operates entirely in an **unsupervised manner**.

---

## 🏗️ Pipeline Architecture

```
Raw Transaction Data
        ↓
Data Validation & Cleaning
        ↓
Customer-Level Aggregation
        ↓
RFM Feature Engineering
        ↓
Scaling & Normalization
        ↓
K-Means Clustering
        ↓
Cluster Validation
        ↓
Business Interpretation
```

This pipeline reflects how unsupervised ML models are structured in **real production workflows**.

---

## 🧠 Model Design

### Why K-Means?

- ✓ **Industry-standard** baseline for customer segmentation
- ✓ **Interpretable** cluster centroids
- ✓ **Scalable** to large datasets
- ✓ **Easy to explain** to non-technical stakeholders

### Feature Engineering

| Feature | Description |
|---------|-------------|
| **Recency** | Days since last purchase |
| **Frequency** | Number of transactions |
| **Monetary** | Total customer spend |

**Minimal but intentional preprocessing** was applied to preserve business signal.

---

## 📈 Evaluation Strategy

Since this is an **unsupervised system**, traditional accuracy metrics are not applicable.

### Evaluation Focus

- ✅ **Silhouette Score** – Cluster separation
- ✅ **Davies–Bouldin Index** – Cluster compactness
- ✅ **Elbow Method** – Optimal number of clusters
- ✅ **Cluster Stability** – Consistency across runs
- ✅ **Business Interpretability** – Actionable segments

> **Note:** Moderate scores are expected and acceptable for real-world customer data.

---

## 🔍 Cluster Interpretation

Clusters are profiled using:
- Mean RFM values
- Relative comparisons across clusters
- Business-driven personas

### Example Customer Segments

| Segment | Profile | Business Action |
|---------|---------|-----------------|
| **High-Value Loyal** | High recency, frequency, monetary | VIP programs, exclusive offers |
| **Recent but Low-Spend** | High recency, low frequency/monetary | Engagement campaigns, cross-sell |
| **Infrequent or At-Risk** | Low recency, low frequency | Re-engagement, win-back campaigns |
| **Frequent Mid-Spenders** | Medium across all metrics | Loyalty programs, upsell |

This mirrors how **analytics teams** convert ML output into **business insights**.

---

## 🛠️ Tech Stack

### Machine Learning
- **Language**: Python
- **Framework**: scikit-learn
- **Algorithm**: K-Means
- **Feature Engineering**: RFM Analysis

### Data Processing
- **NumPy** – Numerical computation
- **Pandas** – Data manipulation

### Environment
- **Google Colab** – For reproducibility and rapid iteration

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- Google Colab account (optional, but recommended)

### Option 1: Run in Google Colab

1. Open the notebook directly in Google Colab
2. Click "Run All" or execute cells sequentially
3. All dependencies are pre-installed in Colab

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/RansiluRanasinghe/Customer-Segmentation-K-Means.git
cd Customer-Segmentation-K-Means

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

The notebook is **fully self-contained** and follows a clear, sectioned workflow.

---

## 📊 Results & Insights

### Optimal Clusters
Analysis using the Elbow Method and Silhouette Score suggests **4-5 customer segments** as optimal.

### Key Findings
- Clear separation between high-value and low-value customers
- Distinct behavioral patterns in purchasing frequency
- Actionable segments for targeted marketing strategies

### Visualization
The notebook includes:
- Cluster distribution plots
- RFM metric comparisons
- 2D/3D cluster visualizations

---

## 📌 Key Learning Outcomes

This project demonstrates:

✔ **End-to-end unsupervised ML pipeline design**  
✔ **Industry-grade data validation practices**  
✔ **Feature engineering for behavioral modeling**  
✔ **Proper evaluation of clustering models**  
✔ **Business-oriented interpretation of ML results**

### Skills Demonstrated
- Unsupervised machine learning
- RFM analysis and customer analytics
- Data preprocessing and feature engineering
- Business-focused ML evaluation
- Production ML thinking

---

## 🔮 Future Improvements

- [ ] Replace RFM features with **embedding-based representations**
- [ ] Add **temporal segmentation** (time-based cohorts)
- [ ] Introduce **model monitoring and drift detection**
- [ ] Convert pipeline into a **reusable production service**
- [ ] Add **API layer** for real-time segmentation
- [ ] Implement **automated reporting** for business stakeholders

---

## 🎯 Use Cases

This segmentation approach can be adapted for:
- **E-commerce** — Personalized marketing campaigns
- **SaaS** — Customer retention strategies
- **Banking** — Credit risk assessment
- **Healthcare** — Patient engagement programs
- **Telecom** — Churn prevention

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Connect

**Ransilu Ranasinghe**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ransilu-ranasinghe-a596792ba)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/RansiluRanasinghe)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:dinisthar@gmail.com)

**Interests:**  
Machine Learning • Unsupervised Learning • Data Engineering • Production ML Systems

Always open to discussions on:
- Customer analytics
- ML system design
- Industry ML best practices

---

<div align="center">

**⭐ If you find this project useful, consider giving it a star!**

**Built with industry ML standards in mind.**

</div>
