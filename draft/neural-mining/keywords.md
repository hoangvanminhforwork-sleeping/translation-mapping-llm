# PRISMA Literature Search Strategy & Keywords

## Search Blocks

### Block 1: Cross-Lingual & Multilingual Language Models (The Domain)
*Broadening from specific "hidden space mapping" to general multilingual representation and embedding spaces.*

* **Core Concepts:**
  * `"cross-lingual representation*" OR "multilingual representation*"`
  * `"cross-lingual transfer" OR "multilingual transfer"`
  * `"shared embedding space*" OR "common embedding space*"`
  * `"multilingual language model*" OR "cross-lingual language model*"`
* **Latent & Hidden Space Terms:**
  * `"latent space*" OR "hidden state*"`
  * `"representation space*" OR "embedding space*"`
  * `"multilingual embedding*"`
* **Broader Synonyms:**
  * `"interlingual representation*" OR "interlingua"`
  * `"cross-lingual alignment" OR "multilingual alignment"`

---

### Block 2: Mechanistic Interpretability, Vector Injection & Steering (The Mechanism)
*Broadening from "activation injection" to include probing, causal analysis, and general intervention methods.*

* **Internal States & Probing:**
  * `"mechanistic interpretability"`
  * `"internal state*" OR "residual stream*"`
  * `"probe*" OR "linear probe*"`
  * `"concept representation*" OR "internal representation*"`
* **Intervention & Steering:**
  * `"activation steering" OR "representation engineering"`
  * `"activation patching" OR "model patching" OR "causal mediation analysis"`
  * `"vector injection" OR "hidden state injection" OR "intervention*"`
  * `"steering vector*" OR "control vector*"`

---

### Block 3: Geometry, Mapping & Optimal Transport (The Mathematical Toolset)
*Broadening from specific "token-level optimal transport" to general geometric alignment and mapping techniques.*

* **Alignment & Mapping:**
  * `"embedding alignment" OR "space alignment"`
  * `"procrustes" OR "orthogonal alignment"`
  * `"manifold alignment" OR "subspace alignment"`
  * `"linear representation hypothesis"`
* **Optimal Transport & Distribution Matching:**
  * `"optimal transport" OR "wasserstein distance"`
  * `"distribution matching" OR "domain adaptation"`
  * `"isomorphic" OR "isometry"`

---

## Master Search Query String

Combine the blocks using the standard PRISMA format: `(Block 1) AND (Block 2) AND (Block 3)`

```text
(
  "cross-lingual representation*" OR "multilingual representation*" OR 
  "cross-lingual transfer" OR "shared embedding space*" OR 
  "multilingual language model*" OR "latent space*" OR "hidden state*" OR 
  "cross-lingual alignment" OR "multilingual alignment"
) 
AND 
(
  "mechanistic interpretability" OR "internal state*" OR "residual stream*" OR 
  "activation steering" OR "representation engineering" OR 
  "activation patching" OR "model patching" OR "causal mediation analysis" OR 
  "vector injection" OR "hidden state injection" OR "steering vector*"
) 
AND 
(
  "embedding alignment" OR "space alignment" OR "procrustes" OR 
  "manifold alignment" OR "linear representation hypothesis" OR 
  "optimal transport" OR "wasserstein distance" OR "isomorphic"
)