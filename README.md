# Credit Risk Scoring with Pandas, SQL, and Machine Learning  

## Overview  
This project uses the **UCI credit card dataset**, which contains **30,000 customer records**. The dataset includes demographic details (age, sex, education, marital status), financial information (credit limits, bill amounts, repayment history), and whether the customer defaulted on their next payment. 

The purpose of this project was to:  
- Demonstrate **parsing and cleaning large datasets** with pandas and SQL.  
- Use **data visualization** to find trends, such as who is more likely to default, who has higher credit limits, and who spends the most.  
- Apply **machine learning** (Random Forest and Logistic Regression) to predict which customers are likely to default and why.  

---

## Sample Data  
Here’s a preview of the dataset:  

![Sample Data Screenshot](./screenshots/sampleData.png)  

---

## Data Analysis with Pandas & SQL  
The first stage focused on **cleaning and exploring** the dataset:  
- Converted numeric codes (e.g., Education = 1,2,3) into **human-readable labels** (Graduate School, University, etc.).  
- Converted amounts from TWD to **Canadian dollars** for easier interpretation.  
- Used **pandas** and **SQL queries** to group and summarize trends.  

### Key Explorations  
- **Average Credit Limit (Canadian)** by education and marital status.  
- **Default Rate** across different groups (education, marital status, sex, age).  
- **Utilization Ratio** (bill ÷ credit limit) distribution.  
- **Top Spenders** by latest bill and by utilization ratio.  

![Average Credit Limit](./screenshots/averageLimit.png)  

From the analysis, clients with a **Graduate School** education have the highest average credit limits in Canadian dollars, followed by those with a University degree.
This could be because individuals with higher education levels often secure higher-paying jobs, leading banks to trust them with larger credit lines.

The data shows that Married clients are slightly more likely to default compared to Single or Unknown groups.
One possible reason could be that married households often have higher combined expenses (mortgages, dependents, etc.), which may increase financial stress and default risk.
