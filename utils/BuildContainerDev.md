# Building and Publishing the Dev Container Images

Two Docker images are maintained in this repo:

| Image tag | Dockerfile | Target hardware |
|---|---|---|
| `xscape-fno4d-dev:cu126` | `Dockerfile.dev` | GH200, H100, A100, RTX 20/30/40xx (sm_75–sm_90) |
| `xscape-fno4d-dev:cu132` | `Dockerfile.dev.blackwell` | All of the above + GB200, GB10 (sm_100) |

---

## Option A — GitHub Actions (recommended)

The workflow file `.github/workflows/docker-dev.yml` builds and pushes both images automatically. Four jobs run in parallel on native runners (no QEMU emulation), then a merge step combines the per-arch images into a single multi-arch manifest.

```
build (cu126-amd64) ─┐
build (cu126-arm64) ─┼─→ merge → jhputschke/xscape-fno4d-dev:cu126
build (cu132-amd64) ─┤
build (cu132-arm64) ─┴─→ merge → jhputschke/xscape-fno4d-dev:cu132
```

### When does it run?

| Event | Builds? |
|---|---|
| Push to `main` that changes `Dockerfile.dev` or `Dockerfile.dev.blackwell` | ✅ Yes |
| Push to `main` that changes only other files | ❌ No |
| Manual trigger ("Run workflow" button in the Actions UI) | ✅ Yes |

### One-time setup

**1. Create a Docker Hub personal access token**

- Log in to [hub.docker.com](https://hub.docker.com)
- Avatar (top right) → **Account Settings** → **Personal access tokens** → **Generate new token**
- Set permissions to **Read & Write**
- Copy the token — it is only shown once

**2. Add secrets to the GitHub repo**

Go to `jhputschke/js-contrib` → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret name | Value |
|---|---|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | the personal access token from step 1 |

**3. Commit and push**

Once the workflow file and secrets are in place, any push to `main` that touches a Dockerfile will trigger the build automatically. You can also click **Actions → Build & push dev images → Run workflow** to trigger manually at any time.

### Runner types

| Runner | Arch | Cost |
|---|---|---|
| `ubuntu-latest` | `linux/amd64` — native | Free tier |
| `ubuntu-24.04-arm` | `linux/arm64` — native | Billed at 2× minute rate |

Using native runners avoids QEMU emulation, which would otherwise extend arm64 build time from ~45 min to 2–3 hours for an image this size.

---

## Option B — Local build on macOS (single arch, no push)

Building a CUDA image locally does **not** require a GPU. `nvcc` is never invoked during `docker build`; the CUDA toolkit is only used at container runtime.

```bash
cd js-contrib/utils

# GH200 / cu126
docker build -t xscape-fno4d-dev:cu126 -f Dockerfile.dev .

# Blackwell / cu132
docker build -t xscape-fno4d-dev:cu132 -f Dockerfile.dev.blackwell .
```

This produces a single-arch image matching the host (arm64 on Apple Silicon, amd64 on Intel Mac). Suitable for local testing; not suitable for distribution to both arch types.

---

## Option C — Local multi-arch build with buildx (slow on Mac)

```bash
# One-time: create a multi-arch builder
docker buildx create --use --name multiarch-builder

# Build and push both arches in one command
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t yourname/xscape-fno4d-dev:cu126 \
  -f Dockerfile.dev \
  --push .
```

> **Warning:** On Apple Silicon, `linux/amd64` runs under QEMU emulation and will take 2–3 hours for this image. Use GitHub Actions (Option A) for multi-arch production builds.

---

## Pulling the image

Once pushed, `docker pull` automatically selects the correct arch:

```bash
docker pull jhputschke/xscape-fno4d-dev:cu126   # GH200 / standard CUDA
docker pull jhputschke/xscape-fno4d-dev:cu132   # Blackwell
```
