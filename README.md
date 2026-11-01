# Adaptive AI Personal Life & Study Coach
### An Agentic and Generative AI System featuring Continuous Personal Behavioral Adaptation
**Track:** Advanced Graduation Project (Option C — Egyptian DEPI)

---

## 🌟 The Core Innovation & USP

Most AI study planners generate a schedule once and reactively reschedule only after a student has already failed or fallen behind. They treat every student identically and use generic algorithms.

This project delivers a **Generative AI Personal Life & Study Coach** where the primary innovation and Unique Selling Proposition (USP) is **Adaptive Behavior driven by a Personal Adaptive Model**:

> *"An AI system that doesn't just create a plan for you, but continuously learns how you work, predicts where you might struggle, and adapts its strategy to help you achieve your goals."*

### The Core 6-Stage Closed Loop
```
Collect Behavior ➔ Understand User ➔ Predict Problems ➔ Adapt Strategy ➔ Execute ➔ Measure ➔ Learn Again
```

---

## 🤖 The 5 Cooperating Specialized Agents

| # | Agent Name | Directory | Primary Role & Core Contribution |
|---|---|---|---|
| **1** | **Generative Learning Model / AI Tutor** | `ai_depi/` | RAG over course materials (`material/*.pdf`), grounded study chat with page citations, personalized explanations adapted to knowledge level, summaries, and active recall flashcards. |
| **2** | **Assessment Agent** | `Abdelrahman_quiz_agent` | Productionized from Abdelrahman's notebooks. Designs diagnostic exams, evaluates answers, classifies errors (Conceptual Gap, Application Error, Factual Slip), isolates weak concepts, and immediately feeds results into the Personal Model. |
| **3** | **Personal Planning Agent** | `personal-planning-agent-main/` | 5-step deterministic schedule optimization, fitting tasks into peak cognitive windows (`09:00-11:00`), fatigue buffers, attention-span task deconstruction, and execution telemetry. |
| **4** | **Goal & Life Coach Agent** | `Goal-Life-Coach-Agent-main/` | Decomposes long-term goals into SMART milestones, tracks daily reflections (mood & energy), calculates burnout risk, manages habit streaks with streak freeze protection (shields) and the comeback halving rule. |
| **5** | **Adaptive Manager Agent** *(Orchestrator & USP)* | `adaptive-manager-agent/` | **The central orchestrator.** Owns the Personal Adaptive Model (`adaptive_model`), runs the Struggle Predictor (Task-miss risk, Topic-struggle risk, Overload risk), switches system strategy modes, and issues multi-agent preemptive interventions. |

---

## 🚀 How to Run the Platform

### Single Command Launch
From the project root:
```bash
python run.py
```

### Access Points
- **Web Application Dashboard:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 🖥️ Web Frontend Architecture ("فرونت لذيذ كده وتظهر عليه النتائج")

Built following the exact design specifications from Section 13 of the project documentation:
- **Design Language:** Calm, premium dark interface (Obsidian slate `#0a0d14` / `#111726`), Electric Violet accents (`#6366f1` / `#8b5cf6`), glassmorphism, responsive layout, and Lucide icons.
- **7 Dedicated Hubs:**
  1. **Dashboard:** Today's agenda, priority tasks, live streak with freeze shields, and real-time AI struggle predictions.
  2. **My Adaptive Model (USP Hub):** Visual gauges for Peak Productivity Window, Effective Focus Duration, Task Delay Factor, Learning Velocity, Burnout Risk, Procrastination Triggers, Topic Mastery, and transparent parameter calibration.
  3. **Smart Planner & Calendar:** University fixed lectures timetable side-by-side with AI study blocks, 1-click schedule optimization, attention-span task splitting, and status toggles.
  4. **AI Study Room (ai_depi):** Course material selector (`Automata 1.pdf`, `Trend AI 1.pdf`), grounded RAG chat with page citations, level-tailored explanations, interactive flip flashcards, and lecture summaries.
  5. **Diagnostic Assessment Center (Abdelrahman Quiz Agent):** Dynamic quiz generator from lecture PDFs, interactive question cards, real-time grading, error classification, and 1-click sync with the Adaptive Model.
  6. **Goals & Accountability (Goal & Life Coach):** SMART milestone roadmaps, daily evening reflection check-in modal (mood & energy slider), habit check-ins, and milestone achievement badges.
  7. **Interactive Simulation Lab:** Pre-configured persona scenarios to demonstrate the closed-loop adapting live for evaluations and presentations.

---

## 📁 System Architecture & What Was Integrated

```
Road Masr proj/
├── backend/
│   ├── app/
│   │   ├── config.py                 # Unified settings & LLM configuration
│   │   ├── database.py               # Unified persistent JSON/SQLite storage
│   │   ├── core_loop.py              # Closed-Loop Multi-Agent Orchestrator
│   │   ├── agents/
│   │   │   ├── tutor_agent.py        # Agent 1 (ai_depi RAG Tutor & Content Generator)
│   │   │   ├── quiz_agent.py         # Agent 2 (Abdelrahman Quiz Agent Pipeline)
│   │   │   ├── planner.py            # Agent 3 (Personal Planning 5-Step Optimizer)
│   │   │   ├── goal_coach.py         # Agent 4 (Goal & Life Coach & Gamification)
│   │   │   └── adaptive_manager.py   # Agent 5 (Adaptive Manager & Struggle Predictor)
│   │   ├── routers/
│   │   │   ├── adaptive.py           # Personal Model, Predictions & Simulation endpoints
│   │   │   ├── planning.py           # Calendar, Optimizer, Tasks & Deconstruction
│   │   │   ├── goals.py              # Goals, Milestones, Reflections & Gamification
│   │   │   ├── assessment.py         # Diagnostic quizzes, evaluation & error analysis
│   │   │   └── tutor.py              # RAG study chat, explanations, flashcards, summaries
│   │   └── main.py                   # Unified FastAPI App mounting all routers & static SPA
├── frontend/
│   ├── index.html                    # Single Page Application (Calm dark theme)
│   ├── css/style.css                 # Glassmorphic responsive dark theme
│   └── js/app.js                     # State management, API client & reactive views
├── material/                         # Course documents (Automata & Trend AI PDFs)
├── data/state_store.json             # Persistent state storage (User, Model, Tasks, Quizzes)
└── run.py                            # Unified application runner
```

---

## 🔒 Optional API Keys & Offline Mode
The entire platform is built with **zero-key offline resilience**:
- All 5 agents operate seamlessly with deterministic algorithms, curated course question banks, document parsers, and local heuristics.
- To enable live cloud LLMs (Groq, Gemini, or OpenAI), simply create a `.env` file or export your API key:
  ```env
  GROQ_API_KEY=your_groq_api_key_here
  GEMINI_API_KEY=your_gemini_api_key_here
  OPENAI_API_KEY=your_openai_api_key_here
  ```
