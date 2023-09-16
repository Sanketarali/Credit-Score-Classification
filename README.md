# Credit-Score-Classification
This repo contains the the task of credit score classification with Machine Learning using Python.


There are three credit scores that banks and credit card companies use to label their customers:

Good
Standard
Poor

# importing the necessary Python libraries and the dataset:
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/77ef1823-d6fa-436b-ad80-c1594698f130)
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/afbe9909-1c05-4393-9bb0-4317a5faf22c)

# Let’s have a look at the information about the columns in the dataset:
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/add55d83-a437-49ac-808d-81806e2c110c)

# Before moving forward, let’s have a look if the dataset has any null values or not:
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/2e7ffa40-acf0-46cf-b663-64d3bc3e6c55)

# The dataset doesn’t have any null values. As this dataset is labelled, let’s have a look at the Credit_Score column values:
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/7629d17e-a48d-4081-a70c-088bd805c6e9)

# Data Exploration
The dataset has many features that can train a Machine Learning model for credit score classification. Let’s explore all the features one by one.

I will start by exploring the occupation feature to know if the occupation of the person affects credit scores:
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/b63d5823-9a14-40d4-9b34-f5e9b02ce50d)
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/65eeeefc-9a41-43a7-8f3a-57a20937c5ad)

<h2>There’s not much difference in the credit scores of all occupations mentioned in the data. Now let’s explore whether the Annual Income of the person impacts your credit scores or not:</h2>


![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/7955b01e-f0a2-409c-ae27-0847cca47f87)
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/4b05ff64-20d3-4eda-b18b-d21cbd2e51b8)
<h2>According to the above visualization, the more you earn annually, the better your credit score is. Now let’s explore whether the monthly in-hand salary impacts credit scores or not:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/946f7ac3-a8a1-4071-8c89-064274c46311)
<h2>Like annual income, the more monthly in-hand salary you earn, the better your credit score will become. Now let’s see if having more bank accounts impacts credit scores or not:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/4e67255a-0e1b-41aa-9fd3-43e3240bd07f)
<h2>Maintaining more than five accounts is not good for having a good credit score. A person should have 2 – 3 bank accounts only. So having more bank accounts doesn’t positively impact credit scores. Now let’s see the impact on credit scores based on the number of credit cards you have:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/84fed054-5efa-4475-95a6-f7586e92a32d)
<h2>Just like the number of bank accounts, having more credit cards will not positively impact your credit scores. Having 3 – 5 credit cards is good for your credit score. Now let’s see the impact on credit scores based on how much average interest you pay on loans and EMIs:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/c5524d90-1bb2-4677-ab53-ff5fc2c079b3)
<h2>If the average interest rate is 4 – 11%, the credit score is good. Having an average interest rate of more than 15% is bad for your credit scores. Now let’s see how many loans you can take at a time for a good credit score:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/e4721c54-b8c0-4f88-aa1a-82192abd0b00)
<h2>To have a good credit score, you should not take more than 1 – 3 loans at a time. Having more than three loans at a time will negatively impact your credit scores. Now let’s see if delaying payments on the due date impacts your credit scores or not:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/749105c0-30a0-4690-b2e8-a242ac80034a)
<h2>So you can delay your credit card payment 5 – 14 days from the due date. Delaying your payments for more than 17 days from the due date will impact your credit scores negatively. Now let’s have a look at if frequently delaying payments will impact credit scores or not:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/10f7e951-1396-4a03-9d7b-97caf0913df8)
<h2>So delaying 4 – 12 payments from the due date will not affect your credit scores. But delaying more than 12 payments from the due date will affect your credit scores negatively. Now let’s see if having more debt will affect credit scores or not:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/bbf76df6-81c3-4081-afde-8759822a2b93)
<h2>An outstanding debt of $380 – $1150 will not affect your credit scores. But always having a debt of more than $1338 will affect your credit scores negatively. Now let’s see if having a high credit utilization ratio will affect credit scores or not:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/362a0dde-88e9-4c81-81cf-7909731473d9)

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/031124a9-349e-4be0-a1e8-231cd1780196)
<h2>So, having a long credit history results in better credit scores. Now let’s see how many EMIs you can have in a month for a good credit score:</h2>

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/c18a30df-1d0a-4151-bd52-53e891f38345)
<h2>The number of EMIs you are paying in a month doesn’t affect much on credit scores. Now let’s see if your monthly investments affect your credit scores or not:</h2>

# Credit Score Classification Model
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/5cef2bc4-d94c-4509-a583-ccdb97ec138f)
![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/ea8fae3d-44fd-4a30-b781-195df6580e6c)

![image](https://github.com/Sanketarali/Credit-Score-Classification/assets/110754364/8c0fecae-c1c2-43a8-9761-b25b2ccea6f9)



