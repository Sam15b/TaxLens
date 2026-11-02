# 10. Agile workflow

You are a team of one, but run the project the way a Deloitte delivery team would: a board, sprints,
issues and reviewed PRs. Together they form the evidence trail for the "Agile" line on the JD.

## 10.1 GitHub Projects board

Create a **Project (board view)** linked to the repo:

```powershell
gh repo create TaxLens --private --source . --push
gh project create --owner "@me" --title "TaxLens"
```

| Column | Meaning | Rule |
|--------|---------|------|
| Backlog | Ideas and future issues | Unsized is fine |
| Sprint Ready | Has acceptance criteria and an estimate | Pull from here at sprint planning |
| In Progress | Branch exists | WIP limit: 2 |
| In Review | PR open, CI running | The PR must link the issue (`Closes #12`) |
| Done | Merged + deployed (or week 1–3: merged) | Automatic when the PR merges |

Custom fields: **Sprint** (iteration, 1 week), **Estimate** (1/2/3/5/8 points), **Area**
(`ingestion`, `retrieval`, `agent`, `api`, `security`, `eval`, `infra`, `docs`).

Labels: `feat`, `bug`, `chore`, `docs`, `eval`, `security`, `stretch`, `good-interview-story`.

## 10.2 Sprint rhythm (1 week)

| Day | Activity | Output |
|-----|----------|--------|
| Monday | Planning: choose issues worth ~20 points from Sprint Ready | Sprint goal written in the iteration description |
| Daily | 3-line stand-up note in a pinned issue: done / next / blocked | A visible history |
| Friday | Demo: record a 1–2 min clip or screenshots of the week's demo | Linked in the sprint issue |
| Friday | Retro: 3 bullets (keep / change / try) | `docs/retros/sprint-N.md` |

## 10.3 Issue template (`.github/ISSUE_TEMPLATE/feature.yml`)

```yaml
name: Feature
description: A unit of work for the board
labels: [feat]
body:
  - type: textarea
    id: why
    attributes: { label: Why, description: "User/business reason" }
    validations: { required: true }
  - type: textarea
    id: what
    attributes: { label: What, description: "Scope; what is explicitly out of scope" }
    validations: { required: true }
  - type: textarea
    id: acceptance
    attributes:
      label: Acceptance criteria
      value: |
        - [ ] ...
        - [ ] Tests added
        - [ ] Docs updated (if behaviour changed)
        - [ ] Eval not regressed (if retrieval/prompt/LLM changed)
    validations: { required: true }
  - type: dropdown
    id: estimate
    attributes: { label: Estimate, options: ["1", "2", "3", "5", "8"] }
```

## 10.4 PR template (`.github/pull_request_template.md`)

```markdown
Closes #

## What changed

## How I tested it
- [ ] Unit tests
- [ ] Integration tests
- [ ] Eval (paste the summary table or link the CI comment)

## Risk / rollback

## Checklist
- [ ] No secrets, real PII or client data in code, fixtures or logs
- [ ] `ruff`, `mypy`, `pytest` pass locally
- [ ] Prompt changes bumped `PROMPT_VERSION`
```

Review your own PR the next morning before merging. Leave at least one review comment on any
non-trivial PR to explain a decision. Those comments read like a design log later.

## 10.5 Definition of done (every issue)

1. The code is merged through a PR that links the issue, and CI is green (including the eval gate
   from week 4 onward).
2. Tests cover the change. Bugs get a regression test.
3. No new `ruff`/`mypy` warnings.
4. Docs in `docs/` are updated if behaviour or setup changed.
5. Changes to the security path (PII, auth, isolation) carry the `security` label and a test.

## 10.6 Sprint 1 backlog (ready to create as issues)

| # | Title | Area | Pts |
|---|-------|------|----:|
| 1 | Scaffold repo: uv, ruff, mypy, pytest, config, `/health` | infra | 2 |
| 2 | docker-compose with Azurite, Phoenix; Pinecone account + `init_pinecone.py` | infra | 2 |
| 3 | Domain models + `Repository` protocol + SQLite repo + Alembic migration | api | 3 |
| 4 | `data/sources.yaml` + `download_sources.py` with sha256 | ingestion | 2 |
| 5 | PDF loader with low-text page detection | ingestion | 2 |
| 6 | Metadata regexes (section, circular, notification, date) + tests | ingestion | 3 |
| 7 | Section-aware chunking with contextual headers + tests | ingestion | 3 |
| 8 | FastEmbed dense + SPLADE sparse + Pinecone store (upsert, hybrid search, section boost) | retrieval | 5 |
| 9 | LLM provider factory (Gemini, Groq, Ollama) with fallback | agent | 2 |
| 10 | Minimal graph: retrieve → generate → check_citations | agent | 3 |
| 11 | `POST /ask` endpoint + schemas | api | 2 |

Total: 29 points. That's ambitious for one week. Move #4 or #9 to sprint 2 if needed, and record
the decision in the retro. Being honest about velocity is part of the process.

## 10.7 Using the history in interviews

- **Board screenshots** with sprints and burn-up (Projects → Insights).
- **One PR the eval gate blocked**, with the CI comment that shows the metric drop. Keep this one on purpose.
- **Retros** that show a decision changed because of data (e.g. "hybrid search added after E1 showed
  vector-only missed section-number questions").
