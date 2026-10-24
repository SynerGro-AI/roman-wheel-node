# Releasing roman-wheel-node

Releases are published to PyPI only by `.github/workflows/release.yml`, from a pushed `v*` tag, through PyPI trusted
publishing (OIDC). No PyPI API token is stored in the repository or its secrets, and none is needed.

## One-time setup (repository owner)

1. **PyPI trusted publisher.** On pypi.org, project `roman-wheel-node` → Publishing → add a GitHub publisher:
   owner `SynerGro-AI`, repository `roman-wheel-node`, workflow `release.yml`, environment `pypi`.
2. **`pypi` environment with a required reviewer.** GitHub → Settings → Environments → `pypi` → enable
   "Required reviewers" and add at least one person (optionally "Prevent self-review"). The publish job checks this
   rule itself and refuses to run without it. (On GitHub Free/Pro/Team, required reviewers are available for public
   repositories only.)
3. Optionally restrict the environment's deployment branches/tags to `v*` tags.

## Cutting a release

1. Merge the release PR into `main`. The version lives only in `src/roman_wheel_node/_version.py`; set the
   `CHANGELOG.md` date for the version.
2. Tag the merge commit on `main` and push the tag: `git tag v1.2.0 <commit-on-main> && git push origin v1.2.0`.
3. The **build** job runs; approve the **publish** job in the Actions UI when it asks for review.

## What the workflow enforces

| Guard | Where | Effect |
|---|---|---|
| Tag commit is on `main` | build, step "Tag commit is on main" (full-history checkout, `fetch-depth: 0`) | `git merge-base --is-ancestor $GITHUB_SHA origin/main`, else the build fails |
| Tag equals package version | build | `v<__version__>` must equal the tag |
| Quality gates | build | `ruff check .`, `pytest -W error`, `python -m build`, `twine check --strict`, fresh-venv wheel install and CLI smoke |
| Reviewed environment | publish, first step | `gh api repos/<repo>/environments/pypi` must list at least one `required_reviewers` protection rule, else the job fails before downloading anything |
| Least privilege | publish | `id-token: write` (OIDC for trusted publishing) and `actions: read` (lets `GITHUB_TOKEN` read the environment's protection rules); workflow default is `contents: read` |
| No half-cancelled uploads | per-job concurrency | build: group `release-build-<ref>`, a newer run cancels an older build; publish: group `publish-<ref>`, `cancel-in-progress: false` |
| Pinned runners and time limits | both jobs | `ubuntu-24.04`; `timeout-minutes` 30 (build) and 15 (publish); every action pinned to a commit SHA |

A manual run (`workflow_dispatch`) runs only the build job; publish runs only for `refs/tags/v*`.

Lint the workflows before changing them: `actionlint` (with `shellcheck` on `PATH`) must report 0 findings.
