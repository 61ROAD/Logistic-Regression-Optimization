# **⚡ Logistic Regression Optimization Engine**

## **📌 Project Overview**

While training machine learning models using high-level APIs is straightforward, understanding and optimizing the underlying mathematical solvers is crucial for scaling models on massive datasets.

This project is a deep dive into the mathematical optimization of **Logistic Regression**. I implemented, tuned, and benchmarked multiple numerical optimization and heuristic algorithms to solve the convex optimization problem of Logistic Regression on a large-scale telecommunications dataset. The core objective was to analyze the trade-offs between computational complexity, convergence speed, and model accuracy.

## **🚀 Optimization Algorithms Implemented**

I systematically explored four distinct optimization paradigms, evolving from classical first/second-order derivatives to evolutionary approaches:

### **1\. Newton's Method (Second-Order Optimization)**

* **Approach:** Utilized both the Gradient and the exact Hessian matrix to find the optimal weights.  
* **Characteristics:** Achieves quadratic convergence rates. Extremely fast in terms of iterations but computationally expensive per step (![][image1]) due to Hessian inversion.  
* **Math Intuition:** ![][image2]

### **2\. Quasi-Newton Method (e.g., BFGS)**

* **Approach:** Approximated the Hessian matrix to bypass the computationally heavy exact inverse calculation.  
* **Characteristics:** Strikes an optimal balance between the memory efficiency of gradient descent and the rapid convergence of Newton's method. Highly effective for large feature spaces.

### **3\. Stochastic Gradient Descent (SGD)**

* **Approach:** A first-order optimization method that updates parameters using a single or a mini-batch of training samples per iteration.  
* **Tuning:** Conducted rigorous hyperparameter tuning (learning rate schedules, momentum, epochs) to escape local minima and ensure stable convergence.  
* **Characteristics:** Highly scalable and memory-efficient, making it the industry standard for massive datasets.

### **4\. Genetic Algorithm (Evolutionary / Heuristic)**

* **Approach:** Framed the weight optimization as a biological evolution process (Selection, Crossover, Mutation).  
* **Characteristics:** A gradient-free optimization strategy. While computationally slower, it is highly robust against getting trapped in local optima and demonstrates the versatility of heuristic solvers in non-convex scenarios.

## **📊 Dataset & Preprocessing**

The model was evaluated on a comprehensive **Telecommunications Customer Dataset**, predicting user behaviors (e.g., premium status, churn).

* **Feature Engineering:** Handled mixed data types including numerical (usage minutes, handset price, income) and categorical (marital status, ethnic, geographic area, dualband) features.  
* **Preprocessing:** Applied rigorous normalization and encoding to ensure numerical stability during Hessian inversion and gradient updates.

## **📈 Benchmarking & Visualization**

To rigorously compare the algorithms, I built an automated evaluation pipeline tracking:

* **Convergence Rate:** Monitored the loss function reduction per iteration.  
* **Execution Time:** Profiled the CPU time required for convergence across different solvers.  
* **Classification Metrics:** Evaluated Accuracy, Precision, Recall, F1-Score, and plotted Confusion Matrices.

*(The execution time comparison and convergence plots are automatically generated and saved directly via the Jupyter Notebook pipeline).*

## **🛠️ Tech Stack**

* **Language:** Python  
* **Interactive Environment:** Jupyter Notebook  
* **Core Computation:** NumPy (Matrix Operations), SciPy (Statistical tools)  
* **Machine Learning & Baselines:** Scikit-Learn (LogisticRegression, SGDClassifier)  
* **Data Visualization:** Matplotlib, Seaborn

## **📁 Project Structure**

.  
├── data\_processed.csv       \# Preprocessed telecommunications dataset ready for modeling  
├── Optimization.ipynb       \# Main Jupyter Notebook containing all algorithms and benchmarks  
└── README.md                \# Project documentation

## **⚙️ Installation & Usage**

1. **Clone the repository:**  
   git clone \[https://github.com/yourusername/Logistic-Regression-Optimization.git\](https://github.com/yourusername/Logistic-Regression-Optimization.git)  
   cd Logistic-Regression-Optimization

2. **Install the required dependencies:**  
   pip install pandas numpy matplotlib seaborn scipy scikit-learn

3. **Launch the Jupyter Notebook:**  
   jupyter notebook Optimization.ipynb

4. Run the cells sequentially to trigger data preprocessing, algorithm training, and dynamic generation of the comparative plots.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAADL0lEQVR4Xu1YPYsTURTNIoLiB4KEKPmYiUYiolWwsbASQSy0cAsF18rCRgsVrG38AYvVYrMWNim1cUGxEQu7tVD8AhHdwhVWWbAIxHMm9w137rxkdnayqXLgkZlz7rvvvvfu+5iUSlNMMUVRBEHwHuUzynq5XN5t9YkiDMNrlisCdOopyjM+NxqNS3juGf2ufs8FOJyDg3dSzlndAjYLqDNr+SKAv1vw+1ie2cG+1judznZwq5rLBCq8QfmD0hRqRrg+ysGEsQAzd5E2licqlcquYJBi38UHg5zRNqj/3GlSPmidALfCTlp+VNsJML/F+SOrEXTiC060fq1W22l5DQT3AHYvaYvnm1ZHoDugfbR8tVqtwf46tJ/D1iC0tXq9ftzyMWSq2bklqznAwWHaIJB7mmewvsAsYPdNtfPP6s1m87T1reHapw+rod4NaF8sHwPiKiuXPLPj0G6390hwnzTPugj+quZ8YAf5C/vX9IMZP6F1cF3OluG+ouNn+ezaR0dPahutMQusRidXJPAnVtNotVp7xS5e1G5GoJW1rQVHnynKZ/weEj/L2gbvf/W7cLS7w2fUO8b3YUtBbM9bPspfiuj9AatpMIXESTyDLm20nQ/sHG3du2vTBSsDlUpz6RQ3vAWUHmI8am0cmCGwWUyQsrAZdGaQsOmK7bzjUP/CRuq69FTv0ZYfSNbI4N3WNnmB+i/g91WCdDOA8iMheACbHm3RqX2O49rbTAcJaTeqi99uVgZlgR0MzP4QbcHSUFIwgH5K7BLnjZsJzVno9aeBeg/FJ/eA1PrLC28H5RBmIyNnEPq6dMQe0JkpCn2ea8nyXH/SNtdj6mDPC3YwlaIirPiCd4C2RN1u4YTb2SyvMWrwoC2zPs8xq+WFd5MhEPh+GcnUIQ/uLUpv2A0i65hAo2dkALZZjeDtY1T9PKAfd2amINe0KA2Dwb3xl4zsZWtrAbs1bjaaU4MWl4bnekZA+225vHAHve+WUxhwfD/wnGGTROZVrSD4tTH0hjEJMIvQ/hHLjw1Iv9lgI58sW4Bw8LmU2j/GjmALPnizsKkP3iIIx/yXRRYwoHOW0/gPXyIPJUH42ckAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,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>