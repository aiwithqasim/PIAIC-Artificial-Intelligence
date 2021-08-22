# Introduction to Amazon SageMaker
## Topics we will be covered in this chapter.
1. Auto managing services by SageMaker
    * Infrastructrue
    * Auto Scaling
    * All ML packages or Dependencies
    * Deployment and so on
2. We will focuse in this chapter
    * Solve pain points faced by ML practioners
    * SageMaker Capabilities
        * Demonstrating the strengths 
    * Setup SageMaker / local machine configeruation.
        * SageMaker **notebook instance**
        * Setting up Amazon **SageMaker Studio**

~~ Code examples included in the book are available on GitHub at https://github.
com/PacktPublishing/Learn-Amazon-SageMaker. You will need to install a Git
client to access them (https://git-scm.com/). ~~   

~~ Chapter 1 PPT is awaliable at : [Chapter 1 Notes](https://docs.google.com/presentation/d/1tA0MQt4ld4SArW5rrIZDP-UKLJozux-dMJoRV1HGI-8/edit?usp=sharing)~~     
----
# Technical requirements
1. [AWS Account](https://aws.amazon.com/getting-started/)
2. [Free Tier](https://aws.amazon.com/free/)
3. Required Installations
    * [AWS CLI](https://aws.amazon.com/cli/)
    * Python 3.x (use anaconda distributors)
        * we will need (Jupyter, pandas, numpy, and more).

## SageMaker Capabilities
* Launched at AWS re:Invent 2017
* [SageMaker Features complete List](https://aws.amazon.com/about-aws/whats-new/machine-learning)
* SageMaker Application Programming Interfaces (**APIs**), and the Software Development Kits (**SDKs**)
* **The main capabilities of Amazon SageMaker**:
    Amazon SageMaker is the ability to build, train, optimize, and deploy
    models on fully managed infrastructure, and at any scale
    * ## Building
    1. SageMaker provides you with two development environments
        * Notebook Instance: EC2, Jupyter, Anaconda and so on.
    2. SageMaker Studio:
        * Full-fledged integrated development environment
    3. 17 Built-in Alogrithms: ML, DL Optimized to run efficiently on AWS. NO ML code to write!
    4. Open source frameworks (Tensorflow, Pytorch, Apache, MXNet, scikit-learn and more)
    5. Your own code running in your own container: Custom ** Python, R, C++, Java and so on**.
    6. [Pre-trained Models ](https://aws.amazon.com/marketplace/solutions/machine-learning).
    7. <span style="color:red">Amazon SageMaker Autopilot</span> uses AutoML to automatically build, train,and optimize models without the need to write a single line of ML code.
    8. Data Prepairing (data pre-processing)
        * Amazon SageMaker Ground Truth
        * Amazon SageMaker Processing: Run data processing and model evaluation batch
        jobs, using either scikit-learn or Spark
    * ## Training
    1. Managed Storage: S3, Amazon EFS or Amazon FSx for Luster depending on your performance requirements
    2. Managed spot training: Using EC2 spot instance for training in order to reduce costs by up to 80%
    3. Distributed Training: Large-Scale training jobs on a cluster of managed instances.
    4. Pip mode: streams infinitely large datasets in S3, Saving the need to copy data around.
    5. <span style="color: red">Automatic model tuning</span>: runs hyperparameters optimization in order ot deliver high-accuracy model more quickly.
    6. Amazon SageMaker Experiments: 
        * tracks
        * Organize
        * compare all sageMaker jobs
    7. Debugger:
        * Capture internal state during training
        * Inspects observe how the model learns
        * Detects unwanted conditions that hurt accuracy.
    * ## Deploying<br>
    Just as with training, Amazon SageMaker takes care of all your deployment infrastructure,
    and brings a slew of additional features:
    1. Real-time endpoints: HTTPS API
        * Server prediction from your model
        * auto-scaling
    2. Batch-Transform: 
        * predict data in batch
    3. CloudWatch
        * Real-time Infra-structure monitering 
    4. Model Monitor:<br>This captures data sent to an endpoint, and
    compares it with a baseline to identify and alert on data quality issues (missing
    features, data drift, and more)
    5. Amazon SageMaker Neo:
        * compile model for specific hardware (edge,sensor)
        * embeded platform and deploys an optimized version using lightweight runtime.
    6. **Amazon Elastic Inference**: This add fractional GPU acceleration to CPU-based instances for best cost ratio for your prediction inftrastructure.
    ----
## The Amazon SageMaker API

Amazon SageMaker is driven by APIs that are implemented in the language SDKs supported by AWS (https://aws.amazon.com/tools/).
   * Python SDK
   * aka the 'SageMaker SDK'
#### The AWS language SDKs
   * Language SDK's implement service-specific API for all AWS S3, EC2 and so on.
   * [SageMaker API](https://docs.aws.amazon.com/sagemaker/latest/dg/api-and-sdk-reference.html)
   * [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html). SageMaker APIs Available in boto3. 
        * This API low level and verbose
        * create_training_job() has a lot of JSON parameters.
        * We don't need to manage Infrastructure as a code with **CloudFormation** this tool use your deveOps team.
        * We use SageMaker SDK instead of **CloudFormation**
   * SageMaker SDK
        * SageMaker SDK is a python SDK specific. [click of details](https://github.com/aws/sagemaker-pythonsdk)
        * [documentation](https://sagemaker.readthedocs.io/en/stable/)
        * <span style="color:red">The code examples in this book are based on the first release of the SageMaker SDK v2</span>
        * Extremely easy and comfortable to fire up a training job (One line of code) and to deploye a model (one line of code). Infrastructure concerns are abstracted away.
        ```
        # Configure the training job
        my_estimator = TensorFlow(
            'my_script.py',
            role=my_sageMaker_role,
            instance_type='ml.p3.2xlarge',
            instance_count=1,
            framework_version='2.1.0')
        # Train the model
        my_estimator.fit('s3://my_bucket/my_training_data/')

        # Deploy the model to an HTTPS endpoint
        my_predictor = my_estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.c5.2xlarge')
        ```


---
# Demonstrating the strengths of Amazon SageMaker
## Alice's and Bob's Problems
   * Solving Alice's problems
      * she is PHD and data Scientist, She is expert in Math and Statistics
      * She focus on data, advancing her research and publishing papers.
      * She don't know much about IT infrastructure, and she honestly doesn't care all for these topics.
      * She work on her desktop workstation
         * She manage her Desktop
         * Install all packages and software
         * When something goes wrong, she wastes precious hours fixing it.
      * She also have remote servers with powerful multi-GPU, connected to petabyte of network-attached storage.
         * Teh team leads meet and try to prioritize projects and workload.
      * ## Let's see how SageMaker and cloud computing can help Alice.
         * Inexpensive SageMaker notebook instance in a minute.
         * Run code own demand on CPU, GPU's and cluster with managing infrastructure.
         * Automatic model tuning feature in SageMaker
         * Deploying models with cople on lines
         * Keeping track of her expenses with AWS console.
## Solving Bob's problems
   * Bob's history
      * He is DevOps engineer,
      * In charge of a large training cluster share by a team of data scientists
         * Setup Auto scaling
         * Capacity planning is still needed to find the right amout of EC2 instances and to optimize the cost using the right mix of reserved.
         * Bob tries to autmatically reduce capacity at night and on weekends when cluster is less busy.
         * Applied **CI/CD**, After validating model
            * dockerize/containers
            * He just hopes that no one will ask him to do PyTorch and Apache MXNet too.
      * ## Let's see how Bob could use SageMaker to improve his ML workflows.
      * Bob could get rid of his bespoke containers and use their built-in counterparts    
      * Migrating the training workloads to SageMaker
      * Bob get rid of his training cluster, and let every data scientist train completely on demand instead  
      * Data Science team would quickly adopt advanced features
         * distributed training
         * Pipe mode
         * Automatic model tuning
---
# <span style="color:darkgreen">Setting up Amazon SageMaker on your local machine</span>
* Install SageMaker Setup in your local machine.
* Use virtualenv
## Install SageMaker SDK with virtualenv
1. Create a new environment named sagemaker, and activate it:
```
$ mkdir workdir
$ cd workdir
$ python3 -m venv sagemaker
```
Activate Envornment in Linux
```
$ source sagemaker/bin/activate
```
OR
Activate Envornment in Windows
```
$ sagemamer\Scripts\activate
(sagemaker) D:\Qasim\PIAIC\Quarter4\Practice\Chapter1\Installation>
```

2. Install boto3
```
pip install boto3 sagemaker pandas
```
3. Now, let's quickly check that we can import these SDKs in Python:
```
$ python3
Python 3.7.4 (default, Aug 13 2019, 15:17:50)
>>> import boto3
>>> import sagemaker
>>> print(boto3.__version__)
1.12.39
>>> print(sagemaker.__version__)
1.55.3
>>> exit
```
4. ### let's create a Jupyter kernel based on our virtual environment:
```
pip install ipykernel
python3 -m ipykernel install --user --name=sagemaker
jupyter notebook
```
5. Creat New
    * Python3
    * sagemaker (select this instance types)
6. Finally, we can check that the SDKs are available, by importing them
7. `deactivate`

## Installing the SageMaker SDK with Anaconda
1. Create virtual envornment with anaconda
```
conda create -y -n conda-sagemaker
conda activate conda-sagemaker
```
2. Install pandas, boto3 and Sagemaker SDKs
```
conda install -y boto3 pandas
pip install sagemaker
```

3. Now, let's add Jupyter and its dependencies to the environment, and create a new kernel:
```
conda install -y jupyter ipykernel
python3 -m ipykernel install --user --name condasagemaker
```
4. Launch jupyter notebook
```
jupyter notebook
```
5. Now, check and verify
```
import boto3
import sagemaker

print(boto3.__version__)
print(sagemaker.__version__)
```
---
# A word about AWS permissions
## Amazon Identity and Access Management (IAM)
1. Create AWS IAM user [IAM User](https://aws.amazon.com/iam)
    * If you're not familiar with IAM at all, please read the following documentation:https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html
2. AWS CLI, eu-west-1 (You can use your nearest region
```
$ aws sagemaker list-endpoints --region eu-west-1
{ 
    "Endpoints": []
}
```
3. For more information on SageMaker permissions, please refer to the documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/security-iam.html.
---
# Setting up an Amazon SageMaker notebook instance
1. Jupyter notebook have already installed Anaconda, numpy, pandas and so on.
    * EC2 Intance
    * GPU and CPU
    * If you're not familiar with S3 at all, please read the following documentation: https://docs.aws.amazon.com/AmazonS3/latest/dev/Welcome.html
2. Login with [AWS Console](https://console.aws.amazon.com/sagemaker/)
    * Select SageMaker Service
    * Find Notebook instance from left panel.
        * create new instance sagemaker jupyter notebook
        * check [sagemaker service prices](https://aws.amazon.com/sagemaker/pricing/)
        * ml.t2.medium select and configure notebook instance setting.
        <img src="<img src="https://github.com/EnggQasim/Sagemaker/blob/main/Chapter1/img/figure1.7.JPG">
3. Permissions and encryption
    * IAM role for S3 to create Amazon SageMaker infrastructure and so on.
    * #### Create an IAM Role
        * S3 buckets you specify - optional
        * click on **Create role**
        * optionally disable root acces to the notebook instance
    * #### Permission and encryption
        * Enable - Give users root access to the notebook        
4. As shown in the following screenshot, the optional Network section lets you pick
the Amazon **Virtual Private Cloud (VPC)** where the instance will be launched.
This is useful when you need tight control over network flows from and to the
instance, for example, to deny it access to the internet. Let's not use this feature here:   
5. Git Repositories
6. Tag, It's always good practice to tag AWS resources, as this makes it much easier to
manage them later on. Let's add a couple of tags.    
7. Five to ten minutes later, the instance is in service, as shown in the following
screenshot. Let's click on Open JupyterLab
<img src="https://github.com/EnggQasim/Sagemaker/blob/main/Chapter1/img/figure1.14.JPG">
 


