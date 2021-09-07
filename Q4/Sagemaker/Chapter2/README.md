# 2 Handling Data Preparation Techniques
## Amazon SageMaker Ground Truth
* build accurate training datasets
## SageMaker Processing
* run your data processing workloads
    * feature engineering
    * data validation
    * model evaluation
    * model interpretation
## Data Analytics Services
* Amazon Elastic Map Reduce
* AWS Glue 
* Amazon Athena
## We will Discuse
* Discovering Amazon SageMaker Ground Truth
* Exploring Amazon SageMaker Processing
* Processing data with other AWS services
___
# Technical requirements
* Need AWS Accounts [Create Account if you have't](https://aws.amazon.com/getting-started/)
* You must familiarize yourself with the [AWS Free Tier](https://aws.amazon.com/free/)
* **Command-Line Interface**  [https://aws.amazon.com/cli/](https://aws.amazon.com/cli/)
* Working Python 3.x environment
    * Anaconda
* All Book codes [ClickHere](https://github.com/PacktPublishing/Learn-Amazon-SageMaker)
___
# Discovering Amazon SageMaker Ground Truth
* Added to Amazon SageMaker in late 2018
* building accurate training datasets
* distribute labeling work to public and private workforces of human labelers
* Built-in workflows and graphical interfaces for common image, video, and text tasks
* Ground Truth can enable automatic
labeling, a technique that trains a machine learning model able to label data without additional human intervention.
* you'll learn how to use Ground Truth to label images and text.
## Using workforces
* Group of workers in charge of labeling data samples.
    * **Amazon Mechanical Turk** [link](https://www.mturk.com/)
        * break down large batch jobs into small work units that can be processed by a distributed workforce
        * we can add 10 or thousands of user globally
        * greatest option for extremly large datasets
            * 100 hourse video
            *  identify other vehicles, pedestrians, road signs, and more
            * If you wanted to annotate every single frame, you'd be looking at 1,000 hours x 3,600 seconds x 24 frames per second = 86.4 million images!
    * **Vendor**
        * Required quality work with specifice domain knowledge
        * AWS has vetted a number of data labeling companies
        * **AWS Marketplace** [https://aws.amazon.com/marketplace/](https://aws.amazon.com/marketplace/), **under Machine
Learning | Data Labeling Services | Amazon SageMaker Ground Truth Services**
    * **Private**
        * Sensitive data can be used by some enternally department workers
        * **Creating a private workforce**
        <img src="https://github.com/EnggQasim/Sagemaker/blob/main/Chapter2/img/private_workforce.JPG">