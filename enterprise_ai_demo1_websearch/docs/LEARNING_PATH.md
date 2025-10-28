# 🧭 Learning Path Map

**Purpose:** Give you a single map from first clone to polished demo. Bookmark this page and check off milestones as you go.

**Timeline:** 2 weeks of active work + Week 3 for demo presentations

---

## 1. Orientation (Day 0)
- ✅ `README.md` – skim to understand the big idea
- ✅ `docs/GETTING_STARTED.md` – run the quick setup to green-light your environment
- ✅ `docs/LEARNING_PATH.md` – you are here; choose your path based on time/experience
- ➕ `docs/STUDENT_GUIDE.md` – optional day-by-day checklist if you like printed agendas
- Optional: Install recommended VS Code extensions (Python, Pylance, Git, Copilot)

**Exit criteria:** Tests pass locally, CLI runs one query, repo cloned under your GitHub account.

---

## 2. Narrative Reading (Days 1–2)
Read the code like a textbook, one chapter at a time. Pair each chapter with its lab and companion tests.

| Chapter | Source File | Companion Tests | Lab | Focus |
|---------|-------------|-----------------|-----|-------|
| 1 | `src/models.py` | `tests/test_models.py` | `docs/labs/LAB_01_MODELS.md` | Data models, exceptions |
| 2 | `src/client.py` | `tests/test_client.py` | `docs/labs/LAB_02_CLIENT.md` | API client, error handling |
| 3 | `src/parser.py` | `tests/test_parser.py` | `docs/labs/LAB_03_PARSER.md` | Data transformation |
| 4 | `src/search_service.py` | `tests/test_search_service.py` | `docs/labs/LAB_04_SERVICE.md` | Orchestration |
| 5 | `src/main.py` | `tests/test_main.py` | `docs/labs/LAB_05_MAIN.md` | CLI design |
| 6 | `src/logging_config.py` | `tests` (logging assertions inline) | `docs/labs/LAB_06_LOGGING.md` | Observability |

**How to work:**
1. Read the chapter commentary.
2. Run the lab tasks (RED → GREEN → REFACTOR) in the linked lab doc.
3. Execute the paired tests to watch coverage protect your changes.
4. Capture a quick reflection (see lab template) before moving on.

**Need a bigger picture?** Skim `docs/architecture.md` before or after Chapters 4–6 to see the system diagram and design decisions.

**Exit criteria:** All six labs complete, reflections logged, you can explain WHY each layer exists.

---

## 3. Practice Loops (Days 3–4)
Strengthen skills with deliberate practice and feedback.

- `docs/TDD_WORKFLOW.md` – rehearse the RED/GREEN/REFACTOR cycle.
- `docs/AI_COLLABORATION.md` – run at least one session with an AI assistant; log the transcript summary.
- `docs/GIT_WORKFLOW.md` – practice atomic commits using the provided checklist.
- Optional stretch cards at the end of each lab – attempt at least one advanced challenge.
- Keep `docs/web_search_openai.md` nearby when working on the client/parser to understand tool behavior and payload structure.

**Exit criteria:**
- You have a short AI-collaboration summary in your notes.
- Your commit history shows purposeful, labeled commits.
- You can troubleshoot test failures without instructor help.

---

## 4. Project Studio (Days 5–9)
Build your own application inspired by the reference architecture.

Core resources:
- `docs/PROJECT_LAUNCH_KIT.md` – architecture sketch template, scope checklist, risk planner.
- `docs/PROJECT_IDEAS.md` – choose or adapt a project concept.
- `docs/openai_tools_research_oct2025.md` – API deep dive for selected tools.
- `docs/AI_COLLABORATION.md` – guidance for pairing with AI during implementation.

Milestones:
1. **Design Brief (Day 5):** Submit template from Launch Kit, including MVP feature list and dependency map.
2. **Architecture Checkpoint (Day 6):** Present model/client/service diagram plus test plan draft.
3. **Implementation Sprint (Days 7–8):** Develop features with TDD; maintain ≥70% coverage.
4. **Internal Demo (Day 9):** Run through CLI/UX with teammates, gather feedback.

**Exit criteria:** Feature-complete MVP, all tests passing, README + setup instructions for your project, coverage report ≥70%.

---

## 5. Demo Prep & Assessment (Days 10–12)
Polish, rehearse, and align with grading rubric.

- `docs/DEMO_PLAYBOOK.md` – 5-minute presentation flow, evidence checklist, rehearsal script.
- `docs/GRADING.md` – ensure rubric boxes checked (API usage, tests, quality, docs/demo).
- Update your project README with install/run instructions and screenshots/log excerpts.
- Record AI collaboration notes and test evidence for submission.

**Exit criteria:**
- Rehearsed demo with timed run-through.
- Slide or console-based presentation ready.
- Submission package: repo link, key transcripts, coverage report, deployment instructions.

---

## 6. Week 3: Demo Presentations
Present your project professionally in your assigned demo slot. While waiting for your turn or after presenting, work quietly on final polish, bug fixes, or documentation improvements. Be respectful of presenters.

---

## 7. Ongoing Growth
After demo presentations, use the reference project as a playground for advanced topics:
- Implement caching or persistence layer.
- Integrate structured logging with a real collector (ELK, Datadog).
- Add CI pipeline mirroring the course tests.
- Explore multi-agent workflows with OpenAI Assistants API.

Document new explorations in a personal learning journal or blog to cement understanding.

---

### Quick Progress Tracker
Check boxes as you advance:

- [ ] Orientation complete
- [ ] Labs 1–6 finished with reflections
- [ ] Practice loops logged (TDD, AI, Git)
- [ ] Project Launch Kit submitted
- [ ] Architecture checkpoint approved
- [ ] MVP built with tests ≥70% coverage
- [ ] Demo rehearsal complete
- [ ] Final submission delivered

Keep this page open throughout the course—it’s your north star.
