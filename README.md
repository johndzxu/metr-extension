# MEASURING AI CAPABILITY ON OLYMPIAD-STYLE MATHEMATICS AND PHYSICS REASONING TASKS

## Students (McGill University)
- Miguel Carrillo Cobián ( miguel.carrillocobian@mail.mcgill.ca )
- De-Jhong Hsu ( de-jhong.hsu@mail.mcgill.ca )
- Tong Wu ( tong.wu7@mail.mcgill.ca )

## Mentors (Mila - Quebec AI Institute)

- Jay Gala ( jay.gala@mila.quebec )
- Fengyuan Liu ( fengyuan.liu@mila.quebec )

## TO DO

Miguel
- Fix Grader -> DONE
- Get accuracy through multiple runs & adjust temperature -> DONE
- Try different "answer_type"
- Get Plots -> Function to convert to json (inspect)

John
- Getting the agent to work automatically

Bill
- SWE Validation

Together - By weekend
- Run Llama family

## Motivation

Recent work by METR Kwa et al. (2025) discusses the task-completion time horizon (the longest human-duration task an AI can complete with fixed success probability) as a metric of AI capability growth. They found that over the six years this horizon has grown exponentially. However, this progress has been measured almost entirely in software domains. To address this gap, we intend to measure AI performance on Olympiad-level mathematics and physics reasoning, mapping success to human-equivalent time, to test if the same scaling law holds beyond programming tasks.

## Setup

To setup, install Inspect AI:
```
pip install inspect-ai
```
and create a `.env` file with your API keys, for example:
```
# .env
INSPECT_EVAL_MODEL=openrouter/minimax/minimax-m2:free
OPENROUTER_API_KEY=[Your OpenRouter API Key]
```

To view log files use:
```
inspect view
```
