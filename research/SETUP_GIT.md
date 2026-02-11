# Git Setup Instructions - MVP-Echo Scribe

This is a **NEW** standalone repository. It was extracted from the `mvp-echo-toolbar` monorepo to be its own project.

## Initial Repository Setup

These commands set up the git repo and push to GitHub for the first time.

### 1. Verify SSH Key

You already have an SSH key at `~/.ssh/id_ed25519.pub`:

```bash
cat ~/.ssh/id_ed25519.pub
```

Output:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEwdSQSWI/mIxFka38J35K0xhFlSPDTzYtEqLD3meDNA corey@t-code.com
```

**Add this key to GitHub** (if not already done):
- Go to https://github.com/settings/keys
- Click "New SSH key"
- Paste the public key above
- Title: "T-Code Server"

### 2. Create GitHub Repo

Go to: **https://github.com/organizations/mvp-scale/repositories/new**

Settings:
- **Repository name**: `mvp-echo-scribe`
- **Description**: `GPU-accelerated transcription service with speaker diarization (NeMo Parakeet + Pyannote)`
- **Visibility**: Public (or Private if preferred)
- **DO NOT** initialize with README, .gitignore, or license (we have these already)
- Click "Create repository"

### 3. Initialize Local Git

```bash
cd /home/corey/projects/mvp-echo-scribe

# Verify we're in clean directory (no .git)
ls -la | grep .git
# Should show nothing

# Initialize
git init
git branch -M main

# Add gitignore (important!)
cat > .gitignore << 'EOF'
# Environment
.env
*.env.local

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/
dist/
build/

# Node
node_modules/
npm-debug.log*

# Docker
.docker-sync/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Models (downloaded at runtime)
models/
*.onnx
*.pt
*.pth

# Temp files
*.tmp
*.log

# Large files
*.mp3
*.wav
*.flac
*.m4a
image.png
EOF

# Stage all files
git add .

# Initial commit
git commit -m "Initial commit: MVP-Echo Scribe transcription service

- GPU transcription with NVIDIA NeMo Parakeet TDT 0.6B
- Speaker diarization with Pyannote 3.1
- Pure-Python torchaudio shim (solves C++ ABI incompatibility)
- FastAPI backend with OpenAI-compatible API
- React frontend with upload/view/export
- Docker deployment on RTX 3090
- Interactive feature mockup (mockup.html)
- Comprehensive roadmap and implementation plan (CURRENT.md)"
```

### 4. Add Remote and Push

```bash
# Add SSH remote (mvp-scale org)
git remote add origin git@github.com:mvp-scale/mvp-echo-scribe.git

# Push to main
git push -u origin main
```

## Ongoing Workflow

After initial setup, use standard git workflow:

```bash
# Check status
git status

# Stage changes
git add backend/api.py frontend/src/App.tsx

# Commit
git commit -m "Add API key authentication middleware"

# Push
git push origin main
```

## Branch Strategy (Optional)

For now, working directly on `main` is fine for POC. Later:

```bash
# Create feature branch
git checkout -b feature/paragraph-detection

# Make changes, commit
git add . && git commit -m "Implement paragraph grouping"

# Push branch
git push -u origin feature/paragraph-detection

# Create PR on GitHub, merge when ready
```

## SSH Config for Multiple GitHub Accounts (If Needed)

If you work with multiple GitHub accounts, create `~/.ssh/config`:

```bash
cat > ~/.ssh/config << 'EOF'
# MVP Scale account
Host github.com-mvpscale
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes

# Other account
Host github.com-other
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_other
  IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
```

Then use:
```bash
git remote add origin git@github.com-mvpscale:mvp-scale/mvp-echo-scribe.git
```

## Verify Setup

```bash
# Test SSH connection
ssh -T git@github.com
# Expected: "Hi <username>! You've successfully authenticated..."

# Check remote
git remote -v
# Expected:
# origin  git@github.com:mvp-scale/mvp-echo-scribe.git (fetch)
# origin  git@github.com:mvp-scale/mvp-echo-scribe.git (push)

# Check branch
git branch -a
# Expected: * main
```

## Troubleshooting Git/SSH

**Permission denied (publickey)**:
```bash
# Test SSH key
ssh -T git@github.com

# If fails, check key added to GitHub:
# https://github.com/settings/keys

# Check SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

**Wrong account/org**:
```bash
# Check which account is authenticated
ssh -T git@github.com
# Shows: "Hi USERNAME! You've successfully authenticated"

# If wrong account, use SSH config aliases (see above)
```

## Files That Should NOT Be Committed

Already in `.gitignore`:
- `.env` - contains HF_TOKEN secret
- `models/` - large model files (downloaded at runtime)
- `__pycache__/`, `node_modules/` - generated files
- `*.mp3`, `*.wav` - test audio files (too large)
- `image.png` - mockup reference screenshot (177KB, not needed in repo)

## Initial Commit Checklist

Before pushing:
- [ ] `.env` file is NOT in git (check with `git status`)
- [ ] `.gitignore` exists and covers secrets
- [ ] `CURRENT.md` is included (has full roadmap)
- [ ] `mockup.html` is included (shows planned features)
- [ ] `README.md` is included (project overview)
- [ ] All backend and frontend code is staged
- [ ] Commit message is descriptive

## Post-Push

After first successful push:

1. Verify on GitHub: https://github.com/mvp-scale/mvp-echo-scribe
2. Check files rendered correctly
3. Add repo description and topics (GPU, transcription, NeMo, Pyannote, FastAPI, React)
4. Consider adding GitHub Actions for Docker builds (optional)
5. Update main `mvp-echo-toolbar` repo to reference this new repo in docs

## Moving Forward

This is now your baseline. All future work happens in this repo:
- `/home/corey/projects/mvp-echo-scribe` (local)
- `https://github.com/mvp-scale/mvp-echo-scribe` (remote)

The old nested location (`mvp-echo-toolbar/mvp-echo-scribe/`) can be removed once you've verified the new repo works.
