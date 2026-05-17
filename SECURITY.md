# Security Policy

**Report a vulnerability privately** — email shauryapunj404@gmail.com or use GitHub's "Security › Report a vulnerability" tab.

## Controls

- CodeQL `security-extended` on push, PR, and weekly schedule.
- Dependabot weekly with `semver-major` ignored.
- Branch protection on `main`: required CodeQL check, linear history, no force-push, no deletion.
