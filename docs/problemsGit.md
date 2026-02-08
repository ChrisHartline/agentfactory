# Git Problems & Solutions — Agent Factory

## What Happened (2026-02-08)

We ran into a cascade of git merge issues while pulling a feature branch (`claude/agent-creation-feature-xZ61A`) into a local `master` branch on Windows.

---

## Problem 1: `index.lock` File Exists

**Error:**
```
error: Unable to create '.git/index.lock': File exists.
Another git process seems to be running in this repository
```

**Why it happened:**
A previous git operation (likely VS Code, Cursor, or a crashed terminal) was interrupted mid-operation, leaving a stale lock file. Git uses `index.lock` to prevent concurrent writes to the index.

**Fix:**
```bash
rm .git/index.lock          # Linux/Mac
del .git\index.lock         # Windows
```

**Prevention:**
- Close other editors/terminals that might be running git operations before pulling
- If using Cursor or VS Code, their built-in git features can hold locks — disable auto-fetch in settings:
  - VS Code/Cursor: `"git.autofetch": false`

---

## Problem 2: Local Changes Would Be Overwritten

**Error:**
```
error: Your local changes to the following files would be overwritten by merge:
    requirements.txt
Please commit your changes or stash them before you merge.
```

**Why it happened:**
Local uncommitted changes to `requirements.txt` conflicted with incoming changes from the feature branch.

**Fix:**
```bash
# Option A: Stash, pull, reapply
git stash
git pull origin <branch>
git stash pop

# Option B: Discard local changes (if they don't matter)
git checkout -- requirements.txt
git pull origin <branch>
```

**Prevention:**
- Always commit or stash before pulling
- Use `git status` before every pull to check for uncommitted work

---

## Problem 3: Unmerged Files / Stuck Merge State

**Error:**
```
error: Pulling is not possible because you have unmerged files.
error: You have not concluded your merge (MERGE_HEAD exists).
```

**Why it happened:**
A merge was started but not completed — the repo was stuck in a half-merged state. This happened because Problem 2 interrupted the merge, leaving conflict markers in files.

**Fix:**
```bash
# If you haven't resolved conflicts yet — abort and start over:
git merge --abort

# If you HAVE resolved conflicts — commit the resolution:
git add -u
git commit -m "Merge <branch>"

# If merge --abort says "no merge to abort" but MERGE_HEAD exists:
git reset HEAD <conflicted-file>
git checkout -- <conflicted-file>
```

**Prevention:**
- When a merge conflict appears, either resolve it fully or abort it — don't leave it hanging
- After resolving conflicts, always `git add` and `git commit` immediately

---

## Problem 4: Nested Git Repo Blocking `git add -A`

**Error:**
```
error: 'agent_factory/' does not have a commit checked out
fatal: adding files failed
```

**Why it happened:**
An old `agent_factory/` directory (pre-restructure code) contained its own `.git` folder, making git treat it as a submodule. `git add -A` tried to add it and failed.

**Fix:**
```bash
# Use -u to only stage tracked files, skipping untracked directories:
git add -u
```

**Prevention:**
- Delete old/unused directories that might contain `.git` folders
- Add them to `.gitignore` if they need to stay
- Avoid `git add -A` or `git add .` in repos with nested git directories

---

## Problem 5: Vim Opens on Every Pull

**Symptom:**
Every `git pull` opens vim asking for a merge commit message.

**Why it happened:**
Pulling a remote branch into a different local branch (e.g., pulling `claude/...` into `master`) creates a merge commit, which requires a message.

**Fix (immediate):**
In vim, type `:wq` and press Enter to save and exit.

**Prevention:**
```bash
# Option A: Use rebase instead of merge (no merge commits)
git pull --rebase origin <branch>

# Option B: Make rebase the default for all pulls
git config --global pull.rebase true

# Option C: Work on the same branch (no cross-branch merging)
git fetch origin <branch>
git checkout <branch>
```

---

## Recommended Git Workflow

### Initial Setup (Do Once)

```bash
# Set rebase as default pull strategy (avoids merge commits and vim)
git config --global pull.rebase true

# Set default editor to something friendlier than vim (pick one)
git config --global core.editor "code --wait"    # VS Code / Cursor
git config --global core.editor "notepad"         # Windows notepad
git config --global core.editor "nano"            # Terminal (Linux/Mac)

# Disable auto-fetch in Cursor/VS Code to avoid lock conflicts
# In settings.json:
#   "git.autofetch": false
```

### Before Starting Work

```bash
# Always check state first
git status

# If clean, fetch and switch to the working branch
git fetch origin
git checkout claude/agent-creation-feature-xZ61A   # or whatever branch

# If you have local changes, stash them
git stash
git checkout <branch>
git stash pop
```

### Pulling Updates

```bash
# Preferred: pull with rebase (clean history, no merge commits)
git pull --rebase origin claude/agent-creation-feature-xZ61A

# If rebase has conflicts:
# 1. Fix the conflicted files
# 2. git add <fixed-files>
# 3. git rebase --continue
# To bail out: git rebase --abort
```

### Quick Reference Card

| Situation | Command |
|---|---|
| Check state before doing anything | `git status` |
| Stale lock file | `del .git\index.lock` |
| Uncommitted changes blocking pull | `git stash` → pull → `git stash pop` |
| Stuck in merge state | `git merge --abort` or `git add -u && git commit` |
| Vim opened unexpectedly | Type `:wq` then Enter |
| Switch to a remote branch | `git fetch origin <branch>` → `git checkout <branch>` |
| Avoid merge commits on pull | `git pull --rebase origin <branch>` |
| See what branch you're on | `git branch` |
| See remote branches | `git branch -r` |

### The Golden Rules

1. **`git status` before every operation.** It takes 1 second and prevents every problem above.
2. **Work on the branch, not across branches.** `git checkout <branch>` instead of pulling a feature branch into master.
3. **Never leave a merge unfinished.** Either complete it (`add` + `commit`) or abort it (`merge --abort`).
4. **Use `git pull --rebase`** (or set it globally) to avoid merge commits and the vim prompt.
5. **Commit or stash before pulling.** Uncommitted changes + pull = conflict guaranteed.
