\# SyMoNeuRaL LoRA Baseline v1



Date: 2025-12-28  

Purpose: Teach Symon factual, non-hallucinatory behavior about SyMoNeuRaL



\## Base Model

Qwen/Qwen2.5-3B-Instruct



\## LoRA Adapter

Directory: my\_qwen\_lora\_baseline\_20251228-001



\## Dataset

symoneural.json (facts + behavior rules)



\## Inference Rules

\- SYSTEM prompt required

\- SYM\_TEMP=0 for factual queries

\- Explicit attention\_mask

\- Chat template enforced

\- Only assistant tokens printed



\## Regression Tests

\- test\_infer.ps1 (brain interface / neurotechnology denial)



\## Definition of Done

\- Regression test passes

\- No hallucinated claims

\- Output matches factual anchors

