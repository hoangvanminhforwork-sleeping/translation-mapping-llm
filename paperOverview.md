# Core objective: Representation mapping between LLM

# Paper Overview

## 1. Project Overview

This project studies machine translation, with a focus on food-related and cultural content.

The main goal is not to build a commercial translation product. The main goal is to study the problem scientifically, build a simple model architecture, and test whether the main ideas work.

The project follows a research-first approach. We want to understand and measure the problems instead of only building a system that seems to work.

The long-term goal is to translate food-related content in a way that helps people from different cultural backgrounds understand unfamiliar food and food-related concepts.

However, the project is still in an early research stage. Some ideas are only proposals and have not been tested yet.

## 2. Core Research Direction

The core technical idea is **Artificial Neural Networks (ANNs) and representation mapping between language models**.

The basic idea is to reuse existing language models instead of training a completely new model from scratch.

For example, we may use one language model for Vietnamese and another for English. Each model processes text and produces internal representations. An ANN learns to map the representation from one model to the representation space of the other model.

The target model then uses its own generation mechanism to produce text in the target language.

The main research questions are:

- Can an ANN learn to map representations between two language models?
- Can the mapped representations be used by the target model to generate meaningful translations?
- How well does this method work compared with a normal translation baseline?
- How much information is preserved or lost during representation mapping?
- Can this approach help with food-related and cultural translation?

These questions should be studied step by step. We should not assume that representation mapping will automatically produce good translations.

## 3. Proposed Model Architecture

### 3.1 General Idea

The initial architecture has three main parts:

1. A source language model.
2. An ANN that maps representations.
3. A target language model that generates the translated text.

A simplified pipeline is:

Source text → Source model → Hidden representation → ANN mapping → Target representation → Target model generation → Translated text

For example, in English-to-Vietnamese translation:

English text → English model → Representation → ANN → Vietnamese representation → Vietnamese model → Vietnamese text

For Vietnamese-to-English translation, the direction is reversed.

The two directions may use separate mapping networks. Whether to share parameters or use another design is still an open question.

### 3.2 Model Architecture Is Not Final

The exact model architecture has not been decided.

Decoder-only language models are one possible choice. Other model architectures may also be considered if they are more suitable for the research goals and available computing resources.

The choice should depend on several factors:

- Whether the model is open-source and its weights are available.
- Whether we can access its internal representations.
- Whether we can control or reuse its text generation process.
- Whether the source and target representations can be mapped in a meaningful way.
- Whether the models can run within our available hardware and training budget.

The core idea remains ANN-based representation mapping. The project should not depend on one specific model family unless experiments show that this choice is necessary.

### 3.3 Which Representations Should Be Mapped?

One initial idea is to map representations from the final hidden layer before the language model's output or generation component.

This is only a starting point, not a fixed decision.

We need to investigate which representation is most suitable for mapping and whether the target model can use the mapped representation correctly.

Possible choices include:

- The final hidden layer.
- An earlier hidden layer.
- A selected group of hidden layers.
- Another representation that can be passed to the target model.

The correct choice must be determined through experiments.

### 3.4 Lightweight Training

An important goal is to keep training as lightweight as possible.

Instead of training two large language models from scratch, we may freeze their parameters and train only the ANN mapping network.

This could reduce training costs and make the project more practical with limited computing resources.

However, freezing the language models is also a design choice that needs to be tested. If the mapping does not work well, other training strategies may be considered.

## 4. Initial Training Idea

The initial training plan uses a parallel bilingual dataset.

Each example contains a sentence in one language and its corresponding translation in the other language.

Let:

- `s_en` be an English sentence.
- `s_vi` be its Vietnamese translation.
- `h_en` be a selected representation produced by the English model.
- `h_vi` be a selected representation produced by the Vietnamese model.
- `f` be the ANN that maps English representations to Vietnamese representations.
- `g` be the ANN that maps Vietnamese representations to English representations.

The basic objective is:

- `f(h_en)` should contain information that allows the Vietnamese model to generate `s_vi`.
- `g(h_vi)` should contain information that allows the English model to generate `s_en`.

An initial idea is to make the mapped representations close to their corresponding target representations.

However, matching vectors alone may not be enough. Two representations can have a small numerical difference but still produce different text.

Therefore, training and evaluation should consider both representation-level metrics and translation-level performance.

The exact loss functions, dataset, model checkpoints, and training procedure are not final.

## 5. Main Technical Problems

Several problems need to be investigated.

### 5.1 The Target Model May Not Understand the Mapped Representation

The target model was not originally trained to receive representations produced by another model.

Even if the mapped vector is numerically close to the target model's native representation, the model may still generate incorrect or meaningless text.

We need experiments to test whether the target model can use these representations successfully.

### 5.2 Representation Similarity Does Not Guarantee Translation Quality

A mapping network may produce vectors that look similar to the target representations without preserving the information needed for translation.

We should evaluate both:

- How close the mapped representations are to the target representations.
- How good the generated translations are.

### 5.3 Model Compatibility

Different models may use different hidden dimensions, tokenization methods, training objectives, and internal representation structures.

The mapping network must handle the relevant differences.

The project should investigate which model combinations are practical and whether some combinations are much harder to align than others.

### 5.4 Computing Resources

The available hardware may limit the size and type of models we can use.

Model selection should consider memory usage, inference speed, training time, and access to model internals.

A smaller experiment that gives reliable evidence is preferable to a large experiment that cannot be completed or evaluated properly.

## 6. Proposed Research Steps

The research can be divided into several stages. These stages are a working plan, not a fixed publication schedule.

### Stage 1: Representation Mapping and Translation

Build a minimal working system using existing language models and an ANN mapping network.

Test whether the target model can generate translations from mapped representations.

Start with a small dataset and simple baselines before increasing the scale.

### Stage 2: Representation Perturbation

Study how changes to a model's internal representation affect the generated text.

Starting from a representation associated with a sentence, apply controlled changes and observe how the output changes.

The goal is to understand the relationship between representation changes and generation behavior.

Possible measurements include:

- Numerical distance from the original representation.
- Changes in generated text.
- Changes in semantic meaning.
- Changes in translation quality.
- The stability of generation under different types of perturbation.

Mean squared error (MSE) and mean absolute error (MAE) may be useful, but they should not be treated as sufficient measures on their own.

### Stage 3: Food-Related Concept Detection

Investigate whether internal representations contain information that helps distinguish food-related concepts from other concepts.

A possible starting point is to use a simple classifier, such as logistic regression, on representations produced by the language models.

Other analysis methods, including representation similarity analysis or clustering, may also be considered.

The goal is to test what information can be recovered from the representations, not to assume that the representations directly encode culture.

### Stage 4: Food Knowledge Retrieval

Investigate whether external food knowledge can help improve translated text.

A possible approach is to build a database of food-related information, encode the entries, and retrieve relevant information when the system encounters a food-related expression.

Retrieval-augmented generation (RAG) is one possible method.

A separate decision mechanism would be needed to determine whether additional information should be inserted into the translation.

The criteria for adding information must be clearly defined and evaluated.

## 7. Separate Research Direction: Survey and Cultural Knowledge

This section describes a related but separate research direction.

It should not be treated as a required component of the core ANN representation-mapping architecture.

### 7.1 Main Question

Can measurable information about food and people's experiences with food help us study some aspects of cultural familiarity?

Culture is a broad and difficult concept. This project cannot assume that culture can be fully represented by a single number or vector.

Instead, we may study specific, measurable concepts related to food and cultural experience.

### 7.2 Familiarity Survey

One possible concept is food familiarity.

Participants may be shown food names and images and asked to rate how familiar each item is to them.

The survey may also ask why an item feels familiar.

Participant information, such as demographic background and relevant food experience, may help explain differences in familiarity.

The survey design must carefully consider confounding factors. For example, familiarity may be related to age, education, travel, migration, personal experience, media exposure, and dietary habits, not only cultural background.

### 7.3 Connecting Survey Results with Food Representations

Food items may also be represented using measurable properties, such as ingredients, biological information, aroma-related information, and ingredient combinations.

We can then investigate whether these representations are associated with participants' reported familiarity or other survey measurements.

A recommendation-system approach is one possible analytical method, especially if the dataset contains participant-item ratings.

However, using a recommendation system does not automatically make the research novel. The research question, dataset, methodology, and evaluation must provide a clear contribution.

### 7.4 What This Research Can and Cannot Show

If the measurements are associated, this may provide evidence that some measurable food properties are related to reported familiarity.

It would not automatically prove that these properties represent culture itself.

The interpretation must remain limited to the concepts measured in the study.

The exact survey variables, hypotheses, participant groups, and analysis methods are still undecided.

## 8. Research Principles

The project should follow these principles:

1. **Keep the core technical question clear.** ANN-based representation mapping is the main direction.
2. **Reuse existing models when practical.** Training large models from scratch is not the default plan.
3. **Keep architecture decisions open.** Decoder-only models are one option, not a requirement.
4. **Test assumptions experimentally.** Do not treat an intuitive explanation as a scientific result.
5. **Use appropriate baselines.** Compare against simpler methods and relevant existing approaches.
6. **Separate correlation from causation.** A relationship between representations and cultural familiarity does not prove that one causes or fully explains the other.
7. **Define measurable claims.** Broad concepts such as culture must be translated into specific research questions and measurable variables.
8. **Keep experiments manageable.** Prefer a sequence of small, reproducible experiments over one overly complicated system.

## 9. Current Decisions and Open Questions

### Current Direction

- The application is machine translation, with a focus on food-related and cultural content.
- The core technical idea is ANN-based representation mapping between language models.
- Existing models may be reused to reduce training costs.
- The mapping network is a central object of study.
- The research should evaluate both internal representations and generated translations.

### Still Undecided

- The exact source and target models.
- Whether the models should be decoder-only or use another architecture.
- Which layers or representations should be mapped.
- Whether the mapping networks should be separate or share parameters.
- The training objective and loss functions.
- The dataset and evaluation benchmarks.
- How to handle the target model's unfamiliarity with mapped representations.
- Which food-related concepts can be detected reliably.
- How external food knowledge should be retrieved and inserted.
- Which survey concepts and statistical methods are appropriate.

These decisions should be made based on feasibility, relevant prior research, and experimental results.

## 10. Guidance for AI Agents

When working on this project, AI agents should:

- Treat this file as a general project overview, not as a final experimental specification.
- Preserve ANN-based representation mapping as the core research direction.
- Avoid assuming that a specific model architecture has already been selected.
- Distinguish confirmed decisions from proposed ideas.
- Identify missing information instead of silently inventing details.
- Check relevant literature before making strong novelty claims.
- Prefer simple, reproducible experiments that test one clear hypothesis at a time.
- Report limitations and negative results honestly.
- Keep the survey-based research direction separate unless a task explicitly connects it to the core architecture.
- Avoid claiming that the system understands or transfers culture unless the experiments provide evidence for that claim.

The immediate priority is to make the representation-mapping idea concrete, implement a small prototype, and determine whether it works before expanding the system toward cultural and food-related translation.
