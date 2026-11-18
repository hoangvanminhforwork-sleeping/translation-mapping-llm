# Objective: Do the entire research

# Document
- My raw thought: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\technicalSummary.md
+ This includes other AI reviews

- The polished version: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\polishedTechnicalSummary.md
+ AI has done the polishing quite good
+ But some parts still have problems - not align with me
+ Ask me

- Paper overview: 
+ Just imagine this architecture: Token in source language => Model of source language => Embedding of the source language => Linear mapping => Embedding of the target language => Model of target language => Token in target language
+ No need to read `paperOverview.md` or `paperOverview02.md` - the document is outdated, moreover, now we will follow linear mapping, not ANN
+ Reasons: There are empirical tests that linear mapping wins more sophisticated method: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\References\2024 - Concept Space Alignment in Multilingual LLMs.pdf

# 1. Create database
- You have to complete the mission M1

- Previous agents could not complete them because of time constraint

- Previous mission link: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\data\mission.md

- Reason: 
+ May provide new insights for culture
+ Database about Vietnamese food for RAG

- You can read the guidance about sending API model in the folder `research-phase01`

# 2. Train a linear mapping for about 8M sentences: Vector injection
- Dataset name is mtet: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\data\data

- Ensure no data leakage:
+ If one file is touched before (because of LoRA or scaled version): ignore them

- Report metrics as normal machine translation paper
+ Just focus on normal machine translation first: Like if we follow this architecture, the BLEU score, chrF, COMET
+ We will focus culture and food later

# 3. Check COMET feasability
- Try to run COMET, smoke test first


# 3. Food and Culture: Data exploratory
## 3.1 Logistic regression probe
- Train a new logistic regression
- By the embedding of Flan-t5 only, ignore Vietnamese model
+ Use the flan-t5 embeding before LoRA

- Dataset is still RecipeNLG
- Number of dishes: 100k total
+ 15k for validation
+ 15k for testing final
+ 70k for training

## 3.2 Download a new decision model
- Model should no more than 1B parameters
- Decision model
- Or you can use the API about decision model
- English (Multilingual model): https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3
- Vietnamese: https://huggingface.co/LakoreAI/PhoLaya-base

## 3.3 Is it food?
- Run both logistic probe and decision model 
- In mtet
- Check how consistent between them
- About the question: "Is this sentence has food name or not" for decision model

- But logistic regression is to check on sliding window, sub-sentences 

# 4. More sophisticated
- Train linear mapping with many couples (x - y) between models
- Report metrics for all parts of the research
(or other, if you have ideas)

# Constraint
- Budget for you: 2$
+ File .env path: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-phase01\.env
+ The variable name: Coding
+ Just ensure you and agent platform does not know the key

- Budget for data worker: $1
+ File .env path: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-phase01\.env
+ The variable name: DataWorker
+ Just ensure you and agent platform does not know the key

- Time: Until 7 p.m Vietnam (please check the datetime now)

- We can negotiate about the scope, time and budget
- Number of sentences is also in scope

# Recommendation
- You can use sub agents
- This mission means you have to do the entire research in phase 1: Translation
+ About "culture" phase, we will do that later
+ This is not prototype anymore

Asking me before start execution
And please read this: "D:\prompt-collection"