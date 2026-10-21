# Course Syllabus

The full syllabus of the course, module by module. Each module heading links to the folder with its labs; the module's own README maps every topic to the lab that covers it.

## Modules

1. [Machine Learning Foundations and Problem Formulation](#module-1-machine-learning-foundations-and-problem-formulation)
2. [Data Preparation and Feature Engineering](#module-2-data-preparation-and-feature-engineering)
3. [Regression and the Mathematics of Learning](#module-3-regression-and-the-mathematics-of-learning)
4. [Classification and Model Evaluation](#module-4-classification-and-model-evaluation)
5. [Trees, Ensembles, Tuning, and Explainable AI](#module-5-trees-ensembles-tuning-and-explainable-ai)
6. [Unsupervised Learning, Recommendations, and Embeddings](#module-6-unsupervised-learning-recommendations-and-embeddings)
7. [Neural Networks and the Road to Generative AI](#module-7-neural-networks-and-the-road-to-generative-ai)
8. [ML-to-GenAI Capstone and Agentic AI Foundations](#module-8-ml-to-genai-capstone-and-agentic-ai-foundations)

## Module 1: Machine Learning Foundations and Problem Formulation

📂 [`Module1_ML_Foundations/`](Module1_ML_Foundations) · [labs and episode plan](Module1_ML_Foundations/README.md)

- What is machine learning?
- Traditional programming vs machine learning vs generative AI
- AI, ML, deep learning, GenAI, and agentic AI
- Types of machine learning:
  - Supervised learning
  - Unsupervised learning
  - Semi-supervised learning
  - Reinforcement learning
- Features, labels, observations, models, and predictions
- Regression, classification, clustering, and recommendation
- Training, validation, and test datasets
- End-to-end ML lifecycle

**Important practical topics**

- Converting a business problem into an ML problem
- Defining inputs, outputs, constraints, and success metrics
- Data leakage
- Correlation vs causation
- Common reasons ML projects fail

**Hands-on lab**

- Explore a real dataset using Pandas
- Identify numerical, categorical, and target variables
- Perform basic exploratory data analysis
- Create training and testing datasets
- Build a first baseline model using scikit-learn

## Module 2: Data Preparation and Feature Engineering

📂 [`Module2_Data_Preparation_Feature_Engineering/`](Module2_Data_Preparation_Feature_Engineering) · [labs and episode plan](Module2_Data_Preparation_Feature_Engineering/README.md)

- Why data quality matters more than algorithm complexity
- Missing data and duplicate records
- Outliers and noisy observations
- Numerical and categorical features
- Scaling and normalization
- Feature engineering and feature selection
- Curse of dimensionality
- ML preprocessing pipelines

**Techniques covered**

- Mean, median, and mode imputation
- One-hot and ordinal encoding
- Standardization and min-max scaling
- Log transformation
- Outlier handling
- Correlation analysis
- Mutual information
- Creating domain-specific features

**Hands-on lab**

- Build a reusable preprocessing pipeline using:
  - Pandas
  - SimpleImputer
  - OneHotEncoder
  - StandardScaler
  - ColumnTransformer
  - Scikit-learn Pipeline

**Connection to GenAI: Explain how feature engineering in classical ML evolves into:**

- Tokenization
- Vector representations
- Embeddings
- Metadata enrichment
- Context preparation for RAG

**Assignment: Prepare a raw dataset and document every preprocessing decision.**

## Module 3: Regression and the Mathematics of Learning

📂 [`Module3_Regression_Mathematics_of_Learning/`](Module3_Regression_Mathematics_of_Learning) · [labs and episode plan](Module3_Regression_Mathematics_of_Learning/README.md)

- Simple and multiple linear regression
- Relationship between features and continuous targets
- Model parameters and coefficients
- Cost and loss functions
- Mean squared error
- Gradient descent intuition
- Learning rate
- Underfitting and overfitting
- Bias–variance trade-off
- Regularization
- Algorithms
- Linear regression
- Ridge regression
- Lasso regression
- Elastic Net
- Decision-tree regression
- Evaluation metrics
- MAE
- MSE
- RMSE
- R2R^2R2
- Adjusted R2R^2R2

**Hands-on lab: Build a house-price or healthcare-cost prediction model:**

- Create a baseline
- Train linear regression
- Apply regularization
- Compare models
- Analyze residuals
- Interpret important features

**Connection to GenAI: Relate regression training to neural-network training:**

- Predictions
- Loss calculation
- Backpropagation
- Gradient-based optimization
- Parameter updates

**Assignment: Compare three regression models and justify the best model using both metrics and business requirements.**

## Module 4: Classification and Model Evaluation

📂 [`Module4_Classification_Model_Evaluation/`](Module4_Classification_Model_Evaluation) · [labs and episode plan](Module4_Classification_Model_Evaluation/README.md)

- Binary and multiclass classification
- Decision boundaries
- Probabilities and thresholds
- Class imbalance
- False positives and false negatives
- Why accuracy can be misleading
- Cost-sensitive model evaluation
- Algorithms
- Logistic regression
- K-nearest neighbours
- Decision tree
- Random forest
- Introduction to gradient boosting
- Evaluation metrics
- Confusion matrix
- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC curve and ROC-AUC
- Precision-recall curve

**Hands-on lab: Build a customer-churn or fraud-classification model:**

- Train multiple classifiers
- Compare evaluation metrics
- Tune the decision threshold
- Handle imbalanced data
- Interpret the confusion matrix

**Connection to GenAI: Classification is used in GenAI systems for:**

- Intent detection
- Prompt routing
- Query classification
- Safety filtering
- Tool selection
- Agent decision-making

**Assignment: Design different threshold strategies for a medical, fraud, or customer-churn scenario.**

## Module 5: Trees, Ensembles, Tuning, and Explainable AI

📂 [`Module5_Trees_Ensembles_Tuning_XAI/`](Module5_Trees_Ensembles_Tuning_XAI) · [labs and episode plan](Module5_Trees_Ensembles_Tuning_XAI/README.md)

- How decision trees make decisions
- Entropy and information gain
- Gini impurity
- Tree depth and pruning
- Bagging vs boosting
- Random forests
- Gradient boosting
- XGBoost overview
- Hyperparameters vs learned parameters
- Model improvement
- Cross-validation
- Grid search
- Random search
- Feature importance
- Error analysis
- Data-centric model improvement
- Explainable AI
- Global vs local explanations
- Feature importance
- Partial dependence
- SHAP overview
- Responsible model interpretation

**Hands-on lab**

- Train a decision tree, random forest, and boosting model
- Use cross-validation
- Tune model hyperparameters
- Visualize a decision tree
- Explain individual predictions
- Perform structured error analysis

**Connection to Agentic AI:**

- Compare decision trees with agent workflows:
  - Conditional branching
  - Rules and learned decisions
  - Router agents
  - Workflow orchestration
  - Confidence-based escalation

**Assignment: Create a model comparison report covering performance, explainability, speed, and deployment complexity.**

## Module 6: Unsupervised Learning, Recommendations, and Embeddings

📂 [`Module6_Unsupervised_Recommendations_Embeddings/`](Module6_Unsupervised_Recommendations_Embeddings) · [labs and episode plan](Module6_Unsupervised_Recommendations_Embeddings/README.md)

- Learning without labelled data
- Clustering and customer segmentation
- Distance and similarity
- Dimensionality reduction
- Anomaly detection
- Recommendation-system fundamentals
- Vector representations and embeddings
- Algorithms and techniques
- K-means clustering
- Hierarchical clustering
- DBSCAN overview
- Principal component analysis
- Isolation Forest overview
- Content-based recommendation
- Collaborative filtering overview
- Cosine similarity

**Hands-on lab**

- Create a semantic recommendation or similarity system:
  - Prepare product or document data
  - Generate feature vectors
  - Calculate cosine similarity
  - Retrieve similar items
  - Visualize clusters using PCA

**Connection to GenAI**

- This module forms the direct bridge to:
  - Text embeddings
  - Semantic search
  - Vector databases
  - Document retrieval
  - RAG systems
  - Agent memory

**Assignment: Build a recommendation or document-similarity prototype and explain how it could become a RAG application.**

## Module 7: Neural Networks and the Road to Generative AI

📂 [`Module7_Neural_Networks_Road_to_GenAI/`](Module7_Neural_Networks_Road_to_GenAI) · [labs and episode plan](Module7_Neural_Networks_Road_to_GenAI/README.md)

- Why classical ML is not sufficient for every problem
- Perceptron and artificial neuron
- Weights, biases, and activation functions
- Forward propagation
- Loss functions
- Backpropagation intuition
- Optimizers and learning rates
- Epochs, batches, and iterations
- Trainable parameters
- Overfitting and dropout
- Deep-learning overview
- Feed-forward neural networks
- CNNs for images
- RNNs and LSTMs for sequences
- Attention mechanism
- Transformer architecture
- Encoder and decoder models
- From ML to LLMs
- Tokenization
- Embeddings
- Self-attention
- Pretraining
- Instruction tuning
- Fine-tuning
- In-context learning
- Temperature and probabilistic generation
- Hallucinations

**Hands-on lab**

- Build a small neural-network classifier, or
- Use a pretrained transformer for sentiment/text classification

**Assignment: Draw and explain the journey from raw text to an LLM-generated response.**

## Module 8: ML-to-GenAI Capstone and Agentic AI Foundations

📂 [`Module8_GenAI_Capstone_Agentic_AI/`](Module8_GenAI_Capstone_Agentic_AI) · [labs and episode plan](Module8_GenAI_Capstone_Agentic_AI/README.md)

**Part 1: Building a reliable GenAI system**

- Foundation models and LLM APIs
- Prompt design fundamentals
- Zero-shot and few-shot prompting
- Structured outputs
- Retrieval-Augmented Generation
- Chunking and document preparation
- Embedding generation
- Vector search
- Retrieval and context construction
- LLM response generation
- Evaluating retrieval and generated answers

**Part 2: Agentic AI foundations**

- What makes an AI system an agent?
- Model, tools, instructions, memory, and state
- Workflow vs agent
- Tool/function calling
- Planning and task decomposition
- Reflection and validation
- Human-in-the-loop approval
- Single-agent vs multi-agent systems
- Agent failure modes:
  - Infinite loops
  - Incorrect tool use
  - Hallucinated actions
  - Excessive autonomy
  - Cost and latency problems
  - Prompt injection

**Hands-on capstone: Build an Intelligent Customer Support Assistant**
