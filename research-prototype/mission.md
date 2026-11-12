# Objective: Complete a prototype of the research

# 1. Download models
a. Vietnamese model: Vit5
- base model: https://huggingface.co/VietAI/vit5-base 
- large model: https://huggingface.co/VietAI/vit5-large
b. English model: Flan T5
- base model: https://huggingface.co/google/flan-t5-base
- large model: https://huggingface.co/google/flan-t5-large

- Why these?
+ Previous models in small-experiment and small-experiment-02 cannot generate (or at least I think so)
+ I think we do not need to choose hard ways
+ We should follow the convention
+ Information about models that can run locally in my laptop: `D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\Local_LLM_runable\Model-Ladder-Benchmark.md`

# 2. Model exploratory
- You know the phrase "Data exploratory" right?
- Yes, do that same to the model and the document
- Create notebooks in `model-exploratory`
+ Markdown should be brief and concise
+ The flow of the notebook should be linear to reduce my cognitive workload
+ Each notebook is about one model

- Why do I need model exploratory?
+ I am a student
+ I do not know much about LLM
+ So you do first, explore the model 
+ In both meaning: General LLM and specific about that model
+ Then I will consume the information
+ Learning style: Scaffolding
+ Consider me as an intern, and you are a senior (surely you perform in tech better than me)
+ And I need you to explore and confirm my intuition...could be brought possible

# 3. Test the instruction-following of the model
- You should read about "Say: {sentence}" and "Nói: {câu}" in my raw note
- Data: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\data\data\test-00000-of-00001.parquet
+ Actually, I have docker, and maybe we have data in database too
+ You could use database to load the data

- Test whether the model follows the instruction or not
+ Number of sentences: 2k
+ Just to check whether the model can copy the prompt sucessfully or not

# 5. Test token generation, and perplexity
- You have data about RecipeNLG and cookbooks in Vietnamese
- Try to write several sentences about the food by yourself (as long as your sentence contains food name)
+ Number of sentences: 100 Vietnamese sentence, 100 english sentence
+ Each sentence contains one dish name
- Check perplexity: Is it high in the dish name?
- Check BLEU: + Yes, you have to generate a small dataset to test my methodology

# Document:
- My raw thought:
+ D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-prototype\thought01.md
+ D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-prototype\system-flow.md

- Document by previous agents:
+ D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\small-experiment-02\agent-work
+ D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\small-experiment\agent-work

- You can read more files, but not recommend

# Principle:
Be transparent about what you cannot do
It is okay that you cannot do everything. But you have to be transparent
If you inform me wrongly about the progress (no matter on purpose or accidentally)
It even costs me more
And ask me, it is normal that I give you missions that conflict

# Negotiable:
- I may assign you the tasks that require more effort than I expects
- We can negotiate about what cannot do
- You have 3 hours and budget is $3 to complete the prototype
+ Because I have to sleep...I have to turn my laptop off
+ I can give you more time and more money, if you think you can run in long night
+ Try not to use anything in C disk, because you may need admin rights
+ And...when I sleep, you need admin rights => Task stuck!

- You can check how much you consumes here: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-prototype\.env
+ Try not to print it out. Try not to leak the key
+ As long as agent platform does not know, you do not know => Fine
+ Name of the variable: THIS_MISSION_API

You can use subagents if you can divide independently, the implementation approach is up to you, I do not care much