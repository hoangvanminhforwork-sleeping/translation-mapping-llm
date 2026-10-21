# Objective: Do the entire research but in small scale

Address of data: `data`

Here is recommended approach and what I want you to do:

# 1. Environment
- Please recreate another .venv, or...download needed libraries
- Create an image for MySQL server through Docker, install pygvector...I do not rememeber the library name
- But that library is to storage the vector of the sentence and the data
- Like....a kind of data loader

# 2. Find suitable LLM
- Read this one to have more information: `Local_LLM_runable`
- I need two models that can be run on my laptop: One native model for English and one native model for Vietnamese
- Don't worry, the model is just run for encoding, not generate tokens
- Find the suitable LLM, the benchmark for this paper is not how many tokens/s, but how many sentence could be encoded/s
- The dimension of the vector between two LLMs should be equal to each other

# 3. Model and Encoding
## 3.1 Encoding, ANN and Hundred of linear regression

### Data loader
- See all files in here: `data\data`
- The .parquet file
- Take random file
- Take random 10k sentences to do small experiment
- Remember Reproducibility rule: The seed for random should be fixed
- Or...after takinng the random, just a folder to contains only the sentence that be taken => Recommended this approach, because different device and configuration may have different outcome even same seed
- Use both LLM to encodes (GPU)
- Update to MySQL server

### Encoding
- For each representation
- Do experiments in both approach
- One ANN for representation
- Or hundred of linear regression for each dimension
+ Which means: each linear regression takes the entire vector of eng sentence => one dimension of vi sentence vector

- ANN and linear regression 

### 3.2 Probing
- See that data: `data\RecipeNLG.csv`
- Take the name of the dish only
- Encode it with English LLM
- Recommend load it with DuckDB because the file is really heavy
- Take about 10k samples only
- The random reproducibility principle stays the same as the #2

- Take 10k samples you just take for encoding in MySQL
- Train a logistic regression to identify: Food or not

Report as the agents.md said