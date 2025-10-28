# 🎤 Demo Playbook – Deliver a 5-Minute Professional Presentation

Use this guide to rehearse and polish your final presentation. Rehearse until you can land everything in 5 minutes without reading from the script.

---

## 1. Demo Structure (5 minutes total)
1. **Opening (30 sec)** – Introduce yourself, project name, user problem.
2. **Architecture Story (90 sec)** – Walk through the layers (models → client → parser → service → interface), highlight adaptations from the reference app.
3. **Testing & Quality (60 sec)** – Show test results, coverage snippet, explain one bug your tests caught.
4. **Live Run (90 sec)** – Execute the CLI/UI with a representative query; narrate what’s happening.
5. **Observability & AI Collaboration (45 sec)** – Share a log snippet or monitoring insight plus one AI pairing takeaway.
6. **Closing (30 sec)** – Summarize impact, future enhancements, thank the audience.

---

## 2. Evidence Checklist
Prepare artifacts ready to screen-share or paste into slides:
- ✅ Architecture diagram or annotated code excerpt
- ✅ Terminal screenshot/video of `pytest` passing + coverage report
- ✅ Log snippet showing contextual data (query, duration, status)
- ✅ Live CLI/UX run (backup recording in case of API failure)
- ✅ AI collaboration summary (bullet list of prompts + outcomes)
- ✅ README highlights (install, run, feature list)

Optional extras: metrics dashboard, performance graph, user feedback quotes.

---

## 3. Rehearsal Template
Fill this out during practice sessions.

| Rehearsal # | Total time | What went well | Needs improvement |
|-------------|------------|----------------|-------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

Aim for at least three rehearsals. Record yourself once to evaluate pacing and clarity.

---

## 4. Script Prompts (Customize)
- **Opening:** “Hi, I’m ____. I built ____ to help ____ solve ____.”
- **Architecture:** “The flow mirrors our reference project: the CLI captures input, `SearchService` enforces policies, `WebSearchClient` calls the ___ API, and `ResponseParser` transforms the payload into this summary.”
- **Testing:** “I practiced strict TDD. Here you see the `test_service_prevents_banned_terms` test fail before I implemented the rule. Our final coverage is 82% with integration tests covering edge cases.”
- **Live Run:** “Let’s see it answer a real question. Behind the scenes the service applies domain filters and logs context—for example you can see the query and duration in `app.log`.”
- **Observability & AI Collaboration:** “Logs helped me catch a malformed response quickly. I collaborated with Claude to brainstorm prompt structures, then validated its suggestions through tests.”
- **Closing:** “In production I’d add caching and a web UI. Thank you—happy to answer questions.”

---

## 5. Handling Demo Risks
| Risk | Mitigation |
|------|------------|
| API downtime / quota exceeded | Pre-recorded fallback video and local mock response |
| CLI typo or crash | Print command cheat sheet, practice slow typing |
| Slow response time | Run warm-up query beforehand, mention expected latency |
| Environment drift | Freeze dependencies, test on clean machine/VM |
| Live debugging request | Prepare “parking lot” response: note it and follow up later |

---

## 6. Submission Package
Before demo day, assemble:
- Link to GitHub repo (main branch clean, README updated)
- Latest coverage report screenshot or HTML export
- Log snippet file (`logs/demo_sample.log`) redacted as needed
- AI collaboration summary (prompts + takeaways)
- Slides or demo outline PDF (if using)

Upload/submit per instructor instructions.

---

## 7. Post-Demo Reflection
Immediately after presenting, jot answers:
- What question from the audience stood out? How did you respond?
- What part of your system impressed the audience most?
- What would you improve if you had one more week?
- Which practice or tool from the course will you carry forward?

Great demos are rehearsed, evidence-backed stories. Use this playbook to make your final presentation confident and memorable.
