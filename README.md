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
Uses a local [Ollama](https://ollama.com) model if reachable (default
`llama3.2`), else OpenAI if `OPENAI_API_KEY` is set, else a deterministic
template so nothing breaks offline.
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

## Faithfulness study (60 players, Ollama `llama3.2`)
| strategy | faithfulness | coverage | hallucination | words |
|----------|:-----------:|:--------:|:-------------:|:-----:|
| template (deterministic) | **0.952** | 0.856 | **0.000** | 46 |
| llm (single agent) | 0.878 | **0.889** | 0.041 | 110 |
| llm + critic (LLM rewrite) | 0.894 | 0.883 | 0.030 | 93 |
| llm + guardrail (SHAP rule) | 0.905 | 0.861 | **0.000** | 103 |

The single LLM produces the richest, highest-coverage explanations but
hallucinates features — almost always inventing *in-game purchases* (11/60). A
second LLM acting as a critic helps only modestly and cannot remove every
hallucination. A lightweight **deterministic guardrail** that drops sentences
citing non-evidence features eliminates hallucination entirely while keeping the
LLM's fluency and coverage — the best trade-off among natural-language methods.
Takeaway: ground the fix in the SHAP evidence, don't throw a second LLM at it.

## Limitations
- The dataset appears synthetic (very clean separability, two dominant features);
  conclusions are about the modelling/XAI method, not real player psychology.
- Faithfulness detection uses transparent alias-based keyword matching, not
  semantic parsing — a deliberate, stated simplification.
