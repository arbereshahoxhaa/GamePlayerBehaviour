# Interpreting ML Models for Video Game Player Behavior using Explainable AI

Bachelor thesis project (UBT, Computer Science and Engineering).

Classic ML on this dataset is everywhere; the contribution here is the layer on
top: a **multi-agent LLM pipeline that turns SHAP attributions into
plain-language retention advice, and then audits itself for faithfulness** to the
model's actual attributions.

## Research questions
1. Which player features drive engagement, and do different XAI methods agree?
2. Can an LLM translate per-player SHAP explanations into useful natural language?
3. **Does adding a Critic agent make those explanations measurably more faithful
   to the SHAP evidence than a single LLM?** (template vs llm vs llm+critic)

## Pipeline
```
raw CSV ─▶ clean/split ─▶ train 3 models ─▶ global XAI (importance/perm/SHAP)
                                   │
                        per-player local SHAP  ──▶  PlayerEvidence
                                   │
            Explainer agent ─▶ narrative ─▶ Critic agent ─▶ revised narrative
                                   │                              │
                                   └──────▶ faithfulness score ◀──┘
```

## Layout
| Path | Purpose |
|------|---------|
| `src/gamexai/data.py` | load, clean, stratified split |
| `src/gamexai/models.py` | Logistic Regression, Decision Tree, Random Forest |
| `src/gamexai/evaluate.py` | CV + held-out metrics, confusion matrices |
| `src/gamexai/xai.py` | global importances + per-player local SHAP (`PlayerEvidence`) |
| `src/gamexai/llm.py` | Ollama / OpenAI / template backend (auto-fallback) |
| `src/gamexai/agents.py` | Explainer agent, Critic agent, orchestrator |
| `src/gamexai/faithfulness.py` | SHAP-faithfulness metric |
| `scripts/01_train.py` | train + evaluate + global XAI |
| `scripts/02_explain.py` | per-player explanations (demo) |
| `scripts/03_faithfulness_eval.py` | the experiment |
| `app.py` | Streamlit dashboard |

## Running
```bash
pip install -r requirements.txt
python scripts/01_train.py              # trains, persists models + figures
python scripts/02_explain.py            # sample narratives
python scripts/03_faithfulness_eval.py --n 60   # faithfulness study
streamlit run app.py                    # dashboard
```

### LLM backend
The client tries a fallback chain and uses the first that works:
**OpenAI** (if a key is set) → **Ollama** (local, free) → **deterministic
template** (always works offline). A failed OpenAI call mid-run drops to Ollama
automatically.

To use OpenAI (better/faster narratives), copy `.env.example` to `.env` and add
your key — `.env` is gitignored:
```bash
cp .env.example .env    # then edit OPENAI_API_KEY=sk-...
```
To use local Ollama instead, just leave `.env` out and run:
```bash
ollama serve
ollama pull llama3.2
```

## Results (held-out test set)
| Model | Accuracy | F1-macro | 5-fold CV F1 |
|-------|----------|----------|--------------|
| Logistic Regression | 0.826 | 0.819 | 0.816 ± 0.003 |
| Decision Tree | 0.874 | 0.870 | 0.868 ± 0.003 |
| Random Forest | 0.903 | 0.898 | 0.894 ± 0.003 |

Engagement is driven overwhelmingly by **sessions per week** and **average
session duration**; demographics and genre are near-irrelevant. All three XAI
methods (impurity, permutation, SHAP) agree.

## Faithfulness study (60 players, two models)
Faithfulness/coverage higher is better; hallucination lower is better.

| strategy | faithfulness | coverage | hallucination | words |
|----------|:-----------:|:--------:|:-------------:|:-----:|
| **llama3.2 (local, ~3B)** | | | | |
| template (deterministic) | 0.952 | 0.856 | 0.000 | 46 |
| llm (single agent) | 0.878 | 0.889 | 0.041 | 110 |
| llm + critic (LLM rewrite) | 0.894 ↑ | 0.883 | 0.030 | 93 |
| llm + guardrail (SHAP rule) | **0.905** | 0.861 | **0.000** | 103 |
| **gpt-4o-mini (OpenAI)** | | | | |
| template (deterministic) | 0.952 | 0.856 | 0.000 | 46 |
| llm (single agent) | 0.917 | 0.978 | 0.030 | 96 |
| llm + critic (LLM rewrite) | 0.868 ↓ | 0.750 | 0.037 | 66 |
| llm + guardrail (SHAP rule) | **0.916** | 0.911 | **0.000** | 92 |

Findings:
- **A stronger model gives better base explanations** (faithfulness 0.88 → 0.92,
  coverage 0.89 → 0.98) but still hallucinates — both models invent the same
  feature, *in-game purchases* (~10/60), so scale does not solve XAI faithfulness.
- **The LLM-critic effect is capacity-dependent**: it *helps* the weak model
  (0.894 ↑) but *hurts* the strong one (0.868 ↓), because when the base output is
  already faithful the critic only over-prunes (coverage 0.98 → 0.75).
- **The deterministic SHAP guardrail is robust across both models**: it removes
  every hallucination (0.000) while preserving coverage and fluency, and beats
  the LLM critic in both regimes. Takeaway: ground the fix in the SHAP evidence,
  don't throw a second (or bigger) LLM at it.

## External validation (transfer to a second dataset)
The dataset-1 models were tested, unchanged, on an **independent, differently
generated** dataset (v3ngen, ~10k rows, real missing values, different class
balance) after cleaning:

| model | Dataset 1 F1 | Dataset 2 F1 | Dataset 2 acc |
|-------|:-----------:|:-----------:|:------------:|
| Logistic Regression | 0.819 | 0.453 | 0.474 |
| Decision Tree | 0.870 | 0.462 | 0.626 |
| Random Forest | 0.898 | **0.487** | **0.633** |

F1-macro drops from ~0.90 to ~0.49: the high in-sample score is **partly an
artefact of dataset 1's synthetic label** (nearly a function of two features).
Random Forest remains the most robust and stays above the majority-class
baseline (F1 ≈ 0.25), so the *sessions/duration signal partially generalises*
while the exact decision boundary does not — a key lesson about trusting
single-dataset accuracy.

## Limitations
- Both datasets are synthetic, so results concern the modelling/XAI method and
  robustness to distribution shift, not real player psychology.
- Faithfulness detection uses transparent alias-based keyword matching, not
  semantic parsing — a deliberate, stated simplification.
- The two datasets share column names but some differ in scale/meaning (e.g.
  PlayTimeHours), which is itself part of the transfer finding.
