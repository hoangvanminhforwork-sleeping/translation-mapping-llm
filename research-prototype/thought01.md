
1. Overview about idea
I have an idea about model for translation

Just view it as DE perspective, we have this:

Token => Model => Layer 1 => Embedding 1 => Layer 2 => Embedding 2....=> Next token

So each layer => Each time the vector is transformed

So...for using one English model, one Vietnamese model assumption like this...I suppose

English token => Model Eng => Embedding at layer X of English model => ANN/Linear regression => Embedding at layer Y of Vietnamese model => Vietnamese token

Same to Vietnamese => English: reverse

English token <= Model Eng <= Embedding at layer X of English model <= ANN/Linear regression <= Embedding at layer Y of Vietnamese model <= Vietnamese token

Which means...we can...remove some layers, we do not need to use all layers

2. Just the final layer of target model

- To ensure the sentence is understanding fully, in the model of the original language, we will take the embedding in the final layer

- Which means, the inference should be: Token => Layer 1 => Layer 2....final layer

- However, because loading two models in the same time in my laptop, quite huge..., I think we can ignore the first part of the target model

- So...Token => Layer 1 of first model => Layer 2 of first model =>....Layer final of first model => ANN/Linear regression => Layer final of second model => Token

3. Loss between layers - pertubation
- Quite interesting, maybe a contribution
- How loss in numbers contribute to loss in BLEU during training
- Just a comment...just an interesting idea comes up
- I just leave it here, until I find something interesting

4. Not translation, just copy
a. Fine tune
- I intend to use small model only
- But they are...not good..even in basic tasks
- So the most basic tasks for them: Copy

- Let the dataset has many sentences s
- Each sentence s has two version: English version s_e and Vietnamese version s_v

- Even "1 + 3 = ?", the model sometimes answers incorrectly
+ It seems like we have to fine tune the model
+ Prompt I am thinking like
    For vietnamese: "Nói: {s_v}" (We call them s_v')
    For English: "Say: {s_e}" (We call them s_e')
+ Model has to reply exactly s_v and s_e
+ Data...maybe one piece is enough: D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\data\data\train-00000-of-00014.parquet

b. So after fine tune sucessfully
- Let's extract embedding vector
- s_v' => e_v', and s_e' => e_e'

- Target: We only train ANN or linear regression
+ e_v' in the final layer => e_e' in the layer (and reverse)

5. Contribution
- Welp, no matter it fails or not...we still have contribution

a. Impact of loss in numbers to BLEU (see the fine tune, we can conduct deeply about that)
- What about more layers?

b. Try to achieve SOTA in translation in small model (no more than 1B for both two models)

c. Bring knowledge to translation
- Which one should be used as "knowledge"?
- See the perplexity? D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\present-Saturday-24th Oct\content.md
...

6. So maybe...our jobs are
- Fine tune model
- Probe food and non food more deeply
- Research about loss in number between layers and neurons to tokens generated




Cross-lingual representation alignment.



Layer-wise knowledge transfer.



Representation learning thông qua paraphrase-equivalent sentences.



Perplexity-based knowledge discovery và dynamic fact insertion.