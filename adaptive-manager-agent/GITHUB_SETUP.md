# Publish Adaptive Manager Agent on GitHub

## 1. Create the repository

1. Sign in to GitHub and select **New repository**.
2. Repository name: `adaptive-manager-agent` (or the team-approved name).
3. Choose **Private** if this is university/team work and the team prefers restricted access.
4. Do not initialize another README, `.gitignore`, or license because this project already contains its own files.
5. Create the repository and copy its HTTPS URL.

## 2. Initialize and push from Windows PowerShell

Open PowerShell in the folder that contains `app`, `tests`, `requirements.txt`, and `README.md`:

```powershell
git init
git add .
git status
git commit -m "Initial commit: Adaptive Manager Agent"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/adaptive-manager-agent.git
git push -u origin main
```

Replace the remote URL with the URL of the repository you created. If Git asks for identity first, configure it:

```powershell
git config --global user.name "Your Name"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

Then repeat the commit and push commands.

## 3. Important before pushing

- Keep `.env.example` in the repository.
- Never commit `.env`, API keys, passwords, or tokens.
- Local SQLite files such as `adaptive_manager.db` are ignored by `.gitignore`.
- `.venv`, Python caches, and pytest caches are also ignored.
- Review `git status` before `git commit`. It should show source files, tests, sample data, requirements, and documentation—not secrets or local runtime files.

## 4. Collaborator access

On GitHub, open **Settings → Collaborators** (or the repository's access-management page) and invite the teammates who need access. They can clone the repository and follow `README.md` to install dependencies and run the service.

## 5. Future updates

After editing project files:

```powershell
git add .
git status
git commit -m "Describe your change"
git push
```
