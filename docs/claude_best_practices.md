# Claude Code Best Practices

Compiled from Anthropic docs, builder.io, humanlayer.dev, and community repos.
Reference this from CLAUDE.md: "Workflow best practices: see /docs/claude_best_practices.md"

---

## 1. Start Every Session Right

- Open the project folder first — Claude's context is tied to the folder
- Run `/init` on a new project to auto-generate a starter CLAUDE.md
- Use **Plan Mode** before any task touching more than 2 files — type `/plan` or switch mode at the bottom of the prompt box
- After the plan is shown, review it before saying yes — this is your quality gate

---

## 2. Context is Everything — Manage it Obsessively

Context degradation is the #1 failure mode in Claude Code.

- **`/clear`** between unrelated tasks — don't let old context pollute new work
- **`/compact`** when context hits ~70% full — summarises history without losing thread
- At 90%+ context: `/clear` is mandatory — responses become unreliable
- One session = one feature. Don't mix concerns in the same session.
- After two failed corrections on the same thing: `/clear` and rewrite the prompt better

---

## 3. Write Prompts That Actually Work

- Be specific: name the file, the function, the expected output
- Good: `"Add input validation to the login form in src/auth/login.py — check for empty fields and invalid email format, show inline errors"`
- Bad: `"Fix the login form"`
- Include constraints: `"Don't change the existing API contract"` or `"Follow the pattern in UserCard.py"`
- Paste errors directly — don't describe them, paste the full stack trace
- Drop screenshots in for UI tasks — cuts iteration cycles significantly

---

## 4. CLAUDE.md Rules

- Keep it under 200 lines — beyond that Claude starts ignoring rules uniformly
- Only put things that apply to EVERY task — not edge case instructions
- Put enforced rules in CLAUDE.md, put explanations/examples in /docs and reference them
- Don't use all-caps "MUST" or "NEVER" thinking it helps — it doesn't reliably
- Wrap domain-specific rules in `<important if="...">` tags for long files
- Avoid auto-generating it from scratch — write it yourself so it stays focused

---

## 5. Git Workflow — Non-Negotiable

- Ask Claude to write commit messages: `"Write a commit message for these changes"`
- Commit often — every time a feature works, commit it
- Use `!git status` inline to check state without leaving context
- Claude can open PRs, write PR descriptions, review diffs — use this

---

## 6. Use Subagents for Research

When you need Claude to explore before building:

```
"Use subagents to investigate how auth is currently handled in this codebase
before suggesting changes"
```

This runs exploration in a separate context window so your main session stays clean for building. Especially useful at the start of a session on an existing project.

---

## 7. Quality Gates — Run These Before Every Commit

Always tell Claude to run these before finishing a task:

1. App runs without errors — no broken imports
2. Lint passes
3. Type check passes (if TypeScript/mypy)
4. Core feature works manually — test it yourself

Add this as a hook or just put it in CLAUDE.md as a non-negotiable final step.

---

## 8. The Prototype Mindset

- Build 20-30 small versions instead of writing long specs first
- Cost of building is low — take many shots
- First version: working code over perfect code
- Second pass: clean it up, add tests, write README
- Third pass: deploy it

---

## 9. Prompts That Work Well

```
# Before building
"Show me a plan. Don't write any code yet."

# When stuck
"Knowing everything you know now, scrap this and implement the elegant solution"

# For exploration
"Use subagents to investigate X and report back — don't make changes yet"

# For quality
"Review this code as a senior engineer — what would you push back on?"

# For interviews/portfolio
"Write a README for this project that explains what it does, 
the technical decisions made, and how to run it locally"
```

---

## 10. What NOT to Do

- Don't give Claude a task and walk away — check the plan first
- Don't pile multiple unrelated tasks in one session
- Don't ignore the plan and just say "go" — review it
- Don't let CLAUDE.md grow to 500 lines — prune it ruthlessly
- Don't skip tests entirely — even 2-3 tests signal engineering quality
- Don't use `--dangerously-skip-permissions` until you understand what Claude can do

---

## 11. For Portfolio Projects Specifically

Every project should have:
- `README.md` — what it is, why it exists, how to run it, tech decisions made
- At least basic tests — shows engineering maturity
- Clean git history — meaningful commits, not "fix fix fix wip wip"
- A deployed link — Vercel, Railway, or Render (all free tier)
- `/docs` folder — roadmap, architecture notes, decisions log

Ask Claude at the end of every project:
```
"Write a README that a hiring manager would find impressive — 
cover what this does, the technical architecture, key decisions, and setup instructions"
```

---

*Sources: Anthropic official docs, builder.io/blog/claude-code, humanlayer.dev/blog, 
github.com/shanraisshan/claude-code-best-practice, rosmur.github.io/claudecode-best-practices*