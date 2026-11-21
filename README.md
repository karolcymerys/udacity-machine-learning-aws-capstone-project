# Udacity Machine Learning - Capstone project

This is solution of capstone projection in Udacity Machine Learning program.

This project aims to solve object counting project for provided images of bins. 
Provided solutions can be utilized in delivery centers for inventory level monitoring.

This project utilizes [Amazon Bin Image Dataset [1]](https://registry.opendata.aws/amazon-bin-imagery/) to propose two solutions:
- pre-trained **ResNet18** model with fine-tuned head adjusted for regression problem
- pre-trained **ResNet101** model with fine-tuned head adjusted for regression problem

This project utilizes **AWS Cloud** and **AWS SageMaker** to perform following steps:
- Preparing datasets
- Training and evaluation **ResNet18** (baseline model)
- Training and evaluation **ResNet101** (target model)

## Try it yourself

### Requirements

- AWS Cloud

Please note that following the steps below will incur costs for your AWS account.

1. Go to AWS Management Console and SageMaker AI
2. Create new SageMaker studio domain with default IAM role
3. Go to IAM
4. Create new policy to *ListBucket* and *GetObject* from `arn:aws:s3:::aft-vbi-pds` and `arn:aws:s3:::aft-vbi-pds/*`
5. Assign newly created IAM policy to default IAM role created for your SageMaker studio domain
6. Open you SageMaker studio domain
7. Run JupyterLab
8. Clone repository `https://github.com/karolcymerys/udacity-machine-learning-aws-capstone-project.git`
9. Open `workspace.ipynb`
10. You're ready to start
11. Once you finish remember to stop started instance like JupyterLab, SageMaker Training Job, SageMaker Endpoint, SageMaker Batch Transform Jobs


1. Execute Jupyter notebook [train_and_deploy.ipynb](./train_and_deploy-solution.ipynb) in SageMaker Studio.

## Project structure

```text
udacity-machine-learning-aws-capstone-project/
├── assets/               # Folder containing screenshots of loss chars from Tensorboard
├── src/                  # Folder with source files
├   ├── inference.py      # Python script used for Batch Transform Job for ResNet-based model (can be used to host SageMaker Endpoint)
├   └── train.py          # Python script used for training ResNet-based model in SageMaker
├── proposal.md           # Proposal document as Markdown file
├── proposal.pdf          # Proposal document as PDF file
├── README.md             # Readme file
├── report.md             # Report document as PDF file
├── report.pdf            # Report document as PDF file
├── requirements.txt      # File containing required Python packages
├── workspace.ipynb       # Workspace as Jupter Notebook
└── workspace.html        # Workspace as HTML file
```
