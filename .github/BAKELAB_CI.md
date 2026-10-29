# BakeLab Linux runner selection

The reusable `_linux-build.yml` and `_linux-test.yml` workflows accept
`runner-labels`, a JSON array such as `'["bb", "cuda"]'`. It overrides the build
runner/prefix or the test matrix's runner labels. Every label must match the
runner; both workflows require this explicit selection in `BakeLab/pytorch`.
Other forks remain disabled. Calls from the `pytorch` organization retain their
existing routing when the input is omitted.

A central workflow can call these workflows from `BakeLab/pytorch` at a reviewed
commit, passing `runner-labels` alongside the existing build/test inputs. On
Gitea, use `self:BakeLab/pytorch/.github/workflows/_linux-build.yml@COMMIT` (and
`_linux-test.yml`) to reference the mirror rather than GitHub. Gitea scoped
workflows run in the consuming repository's context, so the repository guard
still applies. No workflow needs to be added to each consuming repository.

Use a separate call for each hardware class. Overriding all shards of a mixed
CUDA/ROCm/ARM matrix with the same labels does not make those jobs compatible.
The runner must provide the selected container image and GPU runtime inside
the BB sandbox.

This change only enables the reusable Linux jobs and runner selection. The
existing `pull.yml`/`trunk.yml` entry points and upstream runner determinator
remain unchanged. GitHub API calls, upstream images/artifact services, CUDA
architecture selection, and mirror event triggers still need separate Gitea
integration work; this is not an end-to-end CUDA CI setup.
