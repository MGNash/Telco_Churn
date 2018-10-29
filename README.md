# Telco_Churn
Predicting and Understanding Customer Churn with Random Forests

A hypothetical telecom provider called Telco wishes to know about factors related to customer retention. They provide data about 7043 customers, including whether 'churned' (i.e. canceled their service and ceased to be Telco customers) within the last month.

To view a report of my methods and results, follow this link: http://htmlpreview.github.com/?https://github.com/MGNash/Telco_Churn/blob/master/telco.html

I used the Telco Customer Churn dataset. This is a sample dataset created by IBM to demonstrate features of the Watson Analytics platform, and is available here on the Watson Analytics Blog: https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/
It can also be found on Kaggle as a public dataset with a bit more description: https://www.kaggle.com/blastchar/telco-customer-churn/

I will be incorporating the permutation variable importance method implemented in the rfpimp package and described here by Terence Parr, Kerem Turgutlu, Christopher Csiszar, and Jeremy Howard (2018):
http://explained.ai/rf-importance/

I will also be using the partial dependence method described in this article by Guy Cafri and Barbara A. Bailey (2016):
http://www.jds-online.com/file_download/531/JDS150802%E6%A0%BC%E5%BC%8F%E6%AD%A3%E7%A2%BA%E7%89%88.pdf

To reproduce my results, first download 'telco.ipynb' from this repository and 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from the Watson Analytics Blog (link above). Change the working directory in the first code chunk to the directory in which the CSV file is located. This code took about an hour to run on a Thinkpad with an Intel Core i7-5600 CPU @ 2.6 GHz with 8GB of RAM. I set random seeds throughout, so the results should be exactly the same.
