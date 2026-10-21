# Security Policy

This is a teaching repository. It runs locally, offline, and it handles no credentials, except an optional Anthropic API key in Module 8, which is read from your environment and never stored.

## Reporting a vulnerability

If you find a security issue (for example, a lab that could leak an API key, or a dependency with a known vulnerability), please **don't open a public issue**. Use GitHub's [private vulnerability reporting](https://github.com/kripasharmaleadership-ai/Machine-Learning-Course/security/advisories/new) instead. You'll get a reply as soon as possible.

## Keep your API key out of the repository

If you try Module 8 with the real Claude API, set `ANTHROPIC_API_KEY` in your shell or use `ant auth login`. Never paste a key into a lab file or commit a `.env` file (`.env` is already in `.gitignore`).
