# Credit Scoring Policy and Socioeconomic Fairness Analysis using LightGBM

## Project Overview
This project simulates the credit allocation process by predicting default risks, applying a cost-sensitive decision threshold, and evaluating the fairness of these decision policies across different socioeconomic groups. The primary objective is to move beyond simple risk scoring and establish a cost-based decision rule, subsequently analyzing its impact on various income segments to identify potential biases.

This research was developed as a graduation thesis for the Industrial Engineering program at Çukurova University.

## Objectives
*   **Risk Prediction:** Develop a robust machine learning model to predict credit default probabilities.
*   **Cost-Sensitive Thresholding:** Determine an optimal decision threshold balancing the cost of false positives (loss cost) and false negatives (opportunity cost).
*   **Fairness Analysis:** Evaluate the accept/reject rates and error types across different income segments.
*   **Scenario Testing:** Test the hypothesis that the socioeconomic bias is driven by the requested credit amount and related variables rather than direct income levels.

## Methodology & Tech Stack
*   **Model:** LightGBM.
*   **Validation:** 10-Fold Cross-Validation for robust Out-of-Fold (OOF) predictions.
*   **Evaluation Metrics:** AUC (0.7916), KS Statistic (0.4410), Brier Score (0.0656) for calibration, and Population Stability Index (PSI = 0.0088).
*   **Explainability:** TreeSHAP for feature importance and dependency analysis.
*   **Dataset:** Home Credit Default Risk dataset (multi-table structure processed into a single matrix).

## Key Findings
1.  **Model Performance:** The LightGBM model demonstrated strong predictive power and stable, well-calibrated probability outputs.
2.  **Cost Optimization:** Based on defined costs ($c_{FN}=20, c_{FP}=1$), the optimal decision threshold was identified as 0.085.
3.  **Socioeconomic Bias:** The single-threshold policy exhibited a clear bias favoring higher-income segments. As income decreases, rejection rates and expected 'good rejection' rates increase, while 'bad acceptance' rates decrease.
4.  **Hypothesis Rejection:** Scenario analysis adjusting the requested credit amount ($\pm20\%$) failed to eliminate the observed socioeconomic trend, rejecting the hypothesis that the bias is solely driven by the loan amount.
5.  **Feature Impact:** SHAP analysis revealed that external source scores (EXT_SOURCE_1, 2, 3) and debt capacity variables (e.g., PAYMENT_RATE) are the primary drivers of the model's decisions, significantly influencing the socioeconomic outcomes.

## Conclusion
The study highlights that applying a uniform cost-based decision threshold across all applicants can lead to unfair outcomes for lower-income groups. It suggests that a multi-threshold policy, tailored to different economic segments, is necessary for fairer credit allocation.

## Repository Structure
*   `credit_scoring_analysis.py`: The complete Python script containing data preprocessing, feature engineering, LightGBM model training, and scenario analysis.
*   `docs/`: The original thesis document (Turkish).
*   `data/`: The dataset used for this project can be accessed here (https://www.kaggle.com/competitions/home-credit-default-risk/overview).

## Author
**Mustafa Cesmeli**
Industrial Engineer
[linkedin.com/in/mustafacesmeli/] | [cesmelimustafa0@gmail.com]
