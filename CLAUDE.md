# Global Preferences

## Who I am
I'm building a portfolio of prototypes to show engineering quality and customer thinking to hiring managers.
Each project should be clean, deployable, and demonstrate real engineering thinking with considerable depth.
Do not overengineer solutiuons, focus on showing value and actual problem solving.

## Code Style
- Python preferred
- Small, modular functions. One responsibility per function.
- Clear variable names over clever ones

## Design System
All prototypes share the same visual identity:
- Dark background (`#0f0f0f`), card surfaces (`#1a1a1a`), accent green (`#a3e635`)
- Fonts: Lora for headings, JetBrains Mono for UI labels and code
- CSS inline in `st.markdown()` — no external stylesheets
- Consistent component pattern: section label → metric cards → result cards → diagnosis block

## Before finishing ANY task
1. Make sure the app actually runs — no broken imports

## Git Rules
- Never commit directly to main unless asked
- Never commit .env files or secrets
- Commit messages: use "feat:", "fix:", "refactor:", "chore:" prefixes
- Write commit messages that explain WHY, not just what changed

## My Workflow
- Always show me a plan before writing code
- Ask if something is unclear rather than guessing
- Prefer working code over perfect code on first pass
- After implementing, tell me what to test manually
- Focus on delivering value, push back if you think this does not make sense

## For docs
- Always keep the README updated
- If there's a `/docs` folder, read it before making decisions — project context lives there

## Workflow Reference
- Claude Code best practices: see /docs/claude_best_practices.md
