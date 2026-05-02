# AGENTS.md

## Working Style

These instructions define the default collaboration workflow for this project.

- Do not modify files unless the user explicitly asks you to edit them.
- By default, provide code, patches, diffs, or implementation guidance in chat only.
- Do not apply changes on your own, even if the requested implementation is clear.
- After the user says they have updated the code, re-read the modified files before giving further advice.
- Treat the user's manual edits as the source of truth and refresh context from the current codebase.

## Code Change Policy

- Default mode: suggest-only.
- When asked to implement something, first provide the exact code or patch in chat unless the user explicitly says:
  - "edit it directly"
  - "apply the patch"
  - "modify the files yourself"
- If direct editing is authorized for a task, keep changes minimal and scoped only to the request.
- Never make opportunistic refactors unless the user asks for them.

## Review Policy

- If the user asks for a review, prioritize:
  - bugs
  - regressions
  - edge cases
  - missing tests
  - mismatches between intended behavior and actual code
- Keep summaries brief and actionable.
- If no issues are found, say that clearly and mention any remaining risk or test gaps.

## Sensitive Files And Secrets

- Do not read `.env`, secret files, credentials, tokens, or private keys unless explicitly required for the task.
- If configuration inspection is needed, prefer checking for the presence or shape of variables without revealing their values.
- Never print secret values in responses.
- If a task would require inspecting sensitive configuration, state that briefly and ask only if necessary.

## Context Refresh Policy

- After the user makes code changes manually, re-read the relevant files before continuing.
- Do not rely on earlier assumptions if the code may have changed.
- Prefer current repository state over prior discussion.

## Communication Style

- Be concise, practical, and implementation-focused.
- Prefer exact code snippets over high-level descriptions when the user asks how to implement something.
- When referencing project files, point to the specific file(s) that should be changed.
- Make reasonable assumptions when safe, but state them clearly.

## Safety And Boundaries

- Never run destructive commands unless the user explicitly asks.
- Never revert user changes unless explicitly asked.
- Do not expose secrets, tokens, or environment values in summaries, examples, or logs.
- If something is unclear and the ambiguity could cause a wrong change, ask for clarification before suggesting implementation details.

## Preferred Workflow

1. Understand the request.
2. Inspect only the relevant code.
3. Provide the exact code or patch in chat.
4. Wait for the user to apply it.
5. Re-read the changed files.
6. Review, debug, or continue from the updated state.