# 🚀 Project Launch Kit

Use this kit to design, scope, and launch your original AI application. Complete it before Day 5 check-in.

---

## 1. Project Snapshot
- **Project name:**
- **One-sentence pitch:**
- **Primary user persona:**
- **Problem statement:**
- **Success metric:** (e.g., “User gets concise summary in <10s”) 

---

## 2. Scope Definition
### Features
| Priority | Feature | User benefit | Acceptance criteria |
|----------|---------|--------------|--------------------|
| P0 | | | |
| P1 | | | |
| P2 | | | |

### Out of Scope (for now)
List features or integrations you intentionally defer.

### Constraints & Risks
- API limits, data availability, hardware, time, etc.
- Mitigation strategies for top 3 risks.

---

## 3. Architecture Sketch
Describe or diagram how your system will mirror/adapt the reference architecture.

```
CLI / UI → Service Layer → Client(s) → External APIs
                ↓
              Parser
                ↓
             Models
```

Questions to answer:
- Will you keep the CLI or build a different interface?
- What new models/dataclasses do you need?
- Which external APIs or tools are involved beyond OpenAI?

Attach diagram or include link.

---

## 4. Data & Prompt Design
- Example queries or inputs the user will supply.
- Desired outputs: format, tone, length.
- Prompt/payload outline for OpenAI (system instruction, user message, tool selection).
- Safe-guarding strategy: how you validate inputs, handle red-team scenarios, respect policy.

---

## 5. Testing Strategy
Fill the table to align with TDD expectations.

| Layer | Test type | Tooling | Notes |
|-------|-----------|---------|-------|
| Models | Unit | `pytest` | |
| Client | Mocked integration | `pytest`, `pytest-mock` | |
| Service | Integration | `pytest`, fixtures | |
| Interface | System/CLI | `pytest`, `capsys` | |
| Extras | (e.g., logging, metrics) | | |

Add measurable coverage target (≥70%) and plan for test data/fixtures.

---

## 6. Timeline & Milestones
| Date | Milestone | Owner | Status |
|------|-----------|-------|--------|
| Day 5 | Project brief approved | | |
| Day 6 | Architecture + test plan review | | |
| Day 7 | Core models & tests complete | | |
| Day 8 | Client/service integration | | |
| Day 9 | Internal demo | | |
| Day 10 | QA sweep + docs | | |
| Day 11 | Rehearsal | | |
| Day 12 | Final demo | | |

---

## 7. Collaboration & Support Plan
- Teammates or accountability partner(s)
- AI assistant strategy (tools, prompts, reflection cadence)
- Instructor checkpoints and questions to clarify early

---

## 8. Resources & References
List docs, tutorials, and code samples you expect to rely on (include links).

---

## 9. Approval Checklist
- [ ] Scope realistic for remaining time
- [ ] Architecture diagram reviewed
- [ ] Test strategy covers all layers
- [ ] Risks acknowledged with mitigations
- [ ] Instructor/instructor-designee sign-off

Submit this completed kit (PDF, Markdown, or shared doc) ahead of the Day 5 review. Update it as plans evolve to keep everyone aligned.
