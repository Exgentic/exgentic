# Releasing Exgentic

## Release model

Exgentic uses Git tags as the single source of truth for released versions.
There is no manually maintained version string in the source tree.

- A release tag must look like `vX.Y.Z`, for example `v0.2.0`
- Package versions are derived from Git tags via `hatch-vcs`
- PyPI publishing runs from GitHub Actions only for pushed `v*` tags
- The publishing workflow verifies that the tagged commit is reachable from `main`
- GitHub Releases are created manually after PyPI publish succeeds

## One-time repository setup

1. In PyPI, create the `exgentic` project if it does not exist yet.
2. In PyPI project settings, add a Trusted Publisher for this GitHub repository.
3. Use these GitHub values when configuring the publisher:
   - Owner: `Exgentic`
   - Repository: `exgentic`
   - Workflow name: `publish-pypi.yml`
   - Environment name: `pypi`
4. In GitHub, keep the `pypi` environment enabled for this workflow if you want environment-level protections.

See the official docs for the exact PyPI setup steps:
- https://docs.pypi.org/trusted-publishers/using-a-publisher/
- https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/

## Release steps

1. Make sure the release commit is already merged to `main`.
2. Update local refs:
   ```bash
   git checkout main
   git pull --ff-only origin main
   ```
3. Create the annotated tag:
   ```bash
   scripts/release.sh 0.2.0
   ```
4. Push the tag:
   ```bash
   git push origin v0.2.0
   ```

Or do steps 3 and 4 in one command:

```bash
scripts/release.sh 0.2.0 --push
```

5. After the PyPI workflow succeeds, create the GitHub Release manually:
   ```bash
   gh release create v0.2.0 --generate-notes --title "v0.2.0"
   ```

## GitHub Release notes

Use generated notes as the base, then edit the release text to keep it short and useful.

The release description should include:

- What changed for users
- Any packaging, CLI, or behavior changes worth calling out
- Any migration or upgrade note if behavior changed
- A short verification note when helpful, for example that the version is on PyPI

Good default structure:

```md
## Summary
- Short user-facing change 1
- Short user-facing change 2

## Notes
- Optional upgrade or compatibility note
```

Avoid:

- Raw internal implementation details unless they affect users
- Huge changelogs pasted into the release body
- Empty releases with only the tag name when there was a meaningful change

## What happens after the tag is pushed

1. GitHub Actions checks out the tagged commit.
2. The workflow confirms that commit belongs to `main`.
3. The package is built from that exact tag.
4. GitHub exchanges its OIDC identity with PyPI using Trusted Publishing.
5. The distribution is uploaded to PyPI.
6. After that succeeds, create the GitHub Release page for the same tag.

## Verifying the release locally

You can inspect the version derived from a tag before pushing:

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
python -m build
```

The built wheel and sdist should report version `0.2.0`.
If you created a test tag by mistake, delete it locally before pushing:

```bash
git tag -d v0.2.0
```
