# BakeLab wheel release

The Gitea-only CUDA check in `.github/workflows/cuda-ci.yml` validates pushes to
`BakeLab/pytorch:trim`. After the build and CUDA tests pass, it dispatches
`BakeLab/breadkernel`'s `guest-assets.yml` on `main`. Breadkernel builds and
publishes the compatible versioned wheel set, including the PyTorch trim
revision available when the build starts.

The dispatch requires the `BREADKERNEL_WORKFLOW_DISPATCH_TOKEN` Actions secret
in the Gitea `BakeLab/pytorch` repository. Configure a dedicated Gitea account
with access only to `BakeLab/breadkernel`, grant only the repository permission
required to dispatch its workflow, create an access token for that account, and
store it as this repository secret. Do not use an organization administrator
token or add the value to GitHub Actions secrets.

The dispatch calls the Gitea Actions API for the fixed `main` ref and workflow
file `guest-assets.yml`; the token is sent only in the authorization header and
is not written to logs.
