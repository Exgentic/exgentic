# Exgentic Security Policy & Responsible Disclosure

## Security Policy

This security policy applies to all public projects under the Exgentic organization on GitHub. We prioritize security and continuously work to safeguard our systems. However, vulnerabilities can still exist. If you identify a security issue, please report it to us so we can address it promptly.

### Security/Bugfix Versions

- Fixes are released either as part of the next minor version (e.g., 1.3.0 → 1.4.0) or as an on-demand patch version (e.g., 1.3.0 → 1.3.1)
- Security fixes are given priority and might be enough to cause a new version to be released

## Reporting a Vulnerability

We encourage responsible disclosure of security vulnerabilities. If you find something suspicious, we encourage and appreciate your report!

### How to Report

Use the "Report a vulnerability" button under the "Security" tab of the [repository](https://github.com/exgentic/exgentic/security). This creates a private communication channel between you and the maintainers.

### Reporting Guidelines

- Provide clear details to help us reproduce and fix the issue quickly
- Include steps to reproduce, potential impact, and any suggested fixes
- Your report will be kept confidential, and your details will not be shared without your consent

### Response Timeline

- We will acknowledge your report within 5 business days
- We will provide an estimated resolution timeline
- We will keep you updated on our progress

### Disclosure Guidelines

- Do not publicly disclose vulnerabilities until we have assessed, resolved, and notified affected users
- If you plan to present your research (e.g., at a conference or in a blog), share a draft with us at least 30 days in advance for review
- Avoid including:
  - Data from any customer projects
  - User/customer information
  - Details about employees, contractors, or partners

We appreciate your efforts in helping us maintain a secure platform and look forward to working together to resolve any issues responsibly.

## Dependency Management & Supply Chain Security

### Version Capping Policy

All direct dependencies in `pyproject.toml` are capped at the next major version (e.g., `litellm>=1.65.0,<2`). This policy limits the blast radius of supply chain attacks by preventing automatic upgrades to arbitrary future versions.

**Why we cap dependencies:**
- **Supply chain attack mitigation**: Malicious packages can be uploaded to PyPI at any time. By capping at major versions, we limit exposure to known version ranges.
- **Controlled upgrades**: Major version bumps require explicit review and testing before adoption.
- **Stability**: Prevents breaking changes from being automatically pulled in.

**Enforcement:**
- A pre-commit hook (`enforce-dependency-caps`) validates that all dependencies have upper bounds.
- CI will fail if any direct dependency lacks an upper bound.
- The hook runs automatically on every commit and in CI.

### Automated Dependency Updates via Renovate

We use Renovate to keep dependencies up to date while maintaining security:

**14-Day Release Age Gate:**
- Renovate is configured with `minimumReleaseAge: 14 days` for all Python dependencies.
- New package versions are not proposed until 2 weeks after their PyPI release.
- This reduces exposure to day-zero malicious uploads and gives the community time to identify compromised packages.

**Major Version Bumps:**
- Renovate uses `rangeStrategy: "bump"` to update both `uv.lock` and the upper bounds in `pyproject.toml` when a new major version is stable.
- Major version PRs require careful review of breaking changes and thorough testing.

### Reviewing Renovate PRs

When reviewing Renovate PRs that update `uv.lock`:

1. **Check the PR description** for the list of updated packages and their version changes.
2. **Review the lockfile diff** to understand what's changing:
   ```bash
   gh pr diff <pr-number> -- uv.lock
   ```
3. **Verify the release age**: Ensure the new version has been available for at least 14 days.
4. **Check for security advisories**: Look for any CVEs or security issues in the changelog.
5. **Review changelogs**: For major updates, read the package's changelog for breaking changes.
6. **Test thoroughly**: Run the full test suite and any relevant integration tests.

### Lockfile Integrity

A `uv-lock --locked` pre-commit hook (added in PR #65) ensures `uv.lock` stays in sync with `pyproject.toml`:
- The hook rejects commits where the lockfile is out of sync.
- This prevents accidental lockfile drift and ensures reproducible builds.
- If the hook fails, run `uv lock` to regenerate the lockfile, review the changes, and commit.

### Incident Response: Malicious Package Detected

If a malicious package version is discovered in our dependencies:

1. **Immediate containment:**
   ```bash
   # Pin the malicious version as excluded in pyproject.toml
   # Example: "litellm>=1.65.0,!=1.82.7,!=1.82.8,<2"
   ```

2. **Rotate credentials:**
   - Assume any secrets or credentials accessible to the compromised environment may be compromised.
   - Rotate API keys, tokens, and passwords that were accessible during the infection window.

3. **Clean infected environments:**
   ```bash
   # Remove all virtual environments
   rm -rf .venv venv .exgentic/
   
   # Reinstall with the patched dependency specification
   uv sync
   ```

4. **Audit for data exfiltration:**
   - Review logs and network traffic for suspicious outbound connections.
   - Check for unauthorized access to systems or data.

5. **Update lockfile:**
   ```bash
   uv lock
   git add uv.lock pyproject.toml
   git commit -m "Pin malicious package version as excluded"
   ```

6. **Notify the team** and document the incident.

### CVE Scanning with uv audit

The `uv audit` command scans dependencies for known CVEs:

```bash
uv audit
```

**Current status:**
- `uv audit` is temporarily unavailable due to the litellm quarantine (versions 1.82.7 and 1.82.8 are excluded).
- Once the quarantine is lifted and a clean version is available, re-enable regular `uv audit` checks.
- Consider adding `uv audit` to CI once it's operational again.

### Best Practices

- **Never commit lockfiles without review**: Always inspect `uv.lock` diffs before committing.
- **Keep dependencies minimal**: Only add dependencies that are truly necessary.
- **Monitor security advisories**: Subscribe to security mailing lists for critical dependencies.
- **Test updates thoroughly**: Don't merge Renovate PRs without running tests.
- **Document exceptions**: If you must exclude a version (e.g., `!=1.82.8`), document why in a comment or commit message.

## Known Vulnerabilities

There are currently no known vulnerabilities.
