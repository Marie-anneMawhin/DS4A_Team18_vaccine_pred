# DS4A Project - Team 18 - Vaccine Acceptance


*Problem Statement*: One or many COVID-19 vaccines will become available to the general public in the next few months and years. Broad acceptance of these vaccines is thought to be fundamental to fight the global pandemic and it would be of interest to predict their uptake by the general public. This project will analyse the question which social and socio-economic factors have had the biggest influence on people’s decision to get the H1N1 and/or the seasonal flu in late 2009 and early 2010 in the US.

*Data*: US census survey in late 2009 and early 2010, e.g. the National 2009 H1N1 Flu Survey. This phone survey asked respondents whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. These additional questions covered their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviours towards mitigating transmission. A better understanding of how these characteristics are associated with personal vaccination patterns can provide guidance for future public health efforts. The data can be found here: [Flu vaccine drivendata](https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/)


Repository structure
---

`Data`  
<p> as csv (training set (labels and features), test set, imputed, imputed and hot encoded, smote balanced)
<br/><br/>

`Graphs`  
- EDA (missingness, distribution, MCA, PCA, correlation matrix, regression)
- Plot of feature importance (gini/coefficients, permutation, SHAP for h1n1 and seasonal with or without doctor recommendations and meta-analysis)
<br/><br/>
  
`Nb`  
Jupyter notebook for both h1n1 and seasonal flu:
- encoding, SMOTE, EDA, PCA, clustering
- RF classifier and feature importance (gini)
- hyperparameter tuning (model_selection, xgb): for logistic regression, support vector machine classifier (SVC), random forest classifier, XGBoost classifier
- feature importance: using gini/coefficients, permutation or SHAP for logistic regression, support vector machine classifier (SVC), random forest classifier, XGBoost classifier
<br/><br/>

`Results`  
details of GridSearchCV for every models
<br/><br/>







