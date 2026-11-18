# Objective: Write code to implement an entire research

You can read here: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-phase01

Basically, in this mission, you have to do..the same like that, but with different models, with small modifications

The codebase is yours actually, feel free to modify it if needed, but I still have to do verification 

# 1. New models
- English: google-t5/t5-small
+ Link HuggingFace: https://huggingface.co/google-t5/t5-small

- Vietnamese: t5-small-vietnamese
+ Link HuggingFace: https://huggingface.co/nbdaaa/t5-small-vietnamese

- Download them

# 2. Check copy input and doing LoRA if needed
- Same dataset as phase01

- But this time, just check whether the model can copy input exactly the input or not
+ Which means the model has to copy exactly the input
+ Ignore "Nói" and "Say", just {s_e} => {s_e}

- Doing LoRA if needed

- Remember:
+ The core objective of the paper is
+ When running model
+ We do not know which the translated sentence should be
+ But we can predict the vector of the translated sentence

- For example
+ English: "I have a cat"
+ We cannot prompt to the model "Tôi có một con mèo"
+ But we can predict the vector of "Tôi có một con mèo"
+ From vector injection => Model generate token => Measure metrics

# 3. Training
## 3.1 Algorithms
- You have to train both Ridge and LightGBM

- Input: Still vector => one dim

- So I guess we have to train 512 linear regressions and lightGBM

- Configurations...up to you

- Why new models:
+ Because yesterday, smaller models win, a model only 76M
+ So I think I should try these small models
+ Beside, small models => generate much more token locally => Can test with higher number of sentence generation

- Why I ask you to do LightGBM?
+ Because the cosine of preivious work is only about 0.7
+ Which means the angle is 41, why 90 means orthogonal, nothing is related
+ I do not think a decoder can have...vector tolerance high like that

## 3.2 Data spliting
- There are many parquet files in D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\data\data

- I intend to split as: 80/10/10
+ 80% for training
+ 10% for validation
+ 10% for testing
+ Ignore the files that have been used for LoRA before

- In phase 1, only 2000 sentences are for test...the number is a little small

- However, consider the text generation ability of the model first
+ Measure how many token/s in my laptop
+ And estimate time

- In the case we cannot utilize all datasets:
+ We still have to do embedding to the entire dataset
+ But we have to choose which parquet file contains most "representative" samples

## 3.3 Data Pruning / Coreset Selection
- The model may not generate text fast enough

- We do not want to wait for too long

- Let's take the parquet file that contains most "representative" samples
+ The selected samples (The selected parquet file) should contain 
+ If we cannot utilize all, at least use one parquet file

- We use both models in part 1 to encode entire dataset
+ so we have npy files
+ or we can use FAISS for database
+ or..other forms, you decide in this kind of implementation
+ I also have docker if you needed
+ prefer simplicity and isolation

- Paper about dataset: "D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\References\2019 - MTet_ Multi-domain Translation for English and Vietnamese.pdf"
- Github about dataset: https://github.com/vietai/mTet
+ Maybe useful for data exploratory

- Recommendation:
+ Use both models to do embedding the entire dataset (or a part of dataset)
+ Use KMeans, set the number of K = number of fields in dataset
+ Consider which parquet file contains highest number of "representative" (samples that are closet to the centroid), highest number of outliers (hard cases, edge cases)

- Remember, the objective is to select which parquet file that are most representatives and diverse
+ You can choose other approaches

- Then split in 80/10/10

# 4. Storage
- The github for this research: https://github.com/hoangvanminhforwork-sleeping/translation-mapping-llm

- Because embedding the entire dataset may require huge storage

- You can up some in github artifacts
+ Then delete them locally

- I meant all files may not used in this mission, this may includes:
+ npy files that are embedding of other models (maybe the folder `finetune-and-lora`)
+ weight of the models

- You can push the code in that github too

- Remember to name the artifacts that are...easy to rememeber and understand
+ Write information about this in the report

# 5. Write paper
- You have to write problem definition into discussion
- You should follow this structure: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-phase01-ver02\structure.md
- And write in here: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-phase01-ver02\paper.md
- Prioritize illustration over text:
+ You can generate SVG as icon, like my previous paper: "D:\HANU_FIT\NCKH\ICOCO\ICOCO IEEE eXPress.pdf". See the overview methodology, there should be icon 
+ Or you can just use Mermaid, using box and text, I quiet prefer this style: "C:\Users\hoang\Downloads\A Linguistically Coherent and Bloom-Aligned BART-GNN.pdf"
+ Or you can ask other AI to generate icon pics to you: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-phase01-ver02\byteDance.txt

- The markdown should have latex on that

- Feel free to write, no need to think much
+ Because your output is not final
+ I will do the rest

# 6. Doing neural mining
- I have a question...here, why linear mapping always win in many empirical test: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\References\2024 - Concept Space Alignment in Multilingual LLMs.pdf
- No specific objective here, just data exploratory or more exactly, model exploratory
- The objective is to try to develop a framework to explore this phenomenon, that can be applied to other models to have insights
- Models: Two models in this case
+ During training and test and generate token, you should have logs about this
+ Then we can conduct research about that

- Write report in here: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-phase01-ver02\neuralMining.md

- Some...thoughts and other AI's feedbacks, you should read:
+ D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\gpt-neural-linear-feedback.md
+ D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\technicalSummary.md
+ D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\polishedTechnicalSummary.md

# 7. Other
- You should read other markdown files to understand clearly the picture
+ May not help much
+ But you may know why do I have to assign you this mission

- Budget:
+ for you: Coding - $3, the variable name is `coding`, just sometimes to check how much you consume
+ for pic generation - $1, the variable name is `picGenerate`, check how much you consume
+ .env file: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-phase01-ver02\.env
+ Ensure that you do not know the real api and agent platform does not know either

- You can use sub-agents if needed

- You should read: "D:\prompt-collection"
+ Rememeber about using time as sensor
+ Check resources and network speed
+ Check temperature of CPU and GPU

- Your time: Until 8 p.m Vietnam

- Can negotiate about the scope

- Try not to touch C disk, because I have other jobs to do, I have to go out
+ You may need admin to approve
+ And...if I am not with my laptop, no one will approve your request => Missions stuck

- Try to interact with my github first, then close applications for resources for training

- Expected interaction:
+ In the first phase: You should read and collect information, then interact with me like web chat UI, ask me, no need to care about my cognitive workload
+ Then, I will leave everything to you. I will go out