#!/bin/bash
# Per-host launcher for the GLM-5.2 Ray cluster on v6e-128.
# Modes:
#   pull  : docker-login both registries + pull vLLM-TPU image and vbar agent.
#   start : (re)start the vbar control agent + the Ray/vLLM 'node' container.
# Role (head/worker) auto-detected from the TPU worker number metadata.
#
# Why vbar: v6e needs the VBAR control agent (slice-coordination daemon on
# :8353) for multi-host JAX. This VM's metadata leaves VBARCONTROL_AGENT_DOCKER_URL
# empty, so we run it ourselves. Chips stay UNBOUND (vbar drives them via PCI
# resource4); libtpu in the container connects to vbar over the host network +
# the /var/run/tpu-plugin UDS. allow_unsafe_interrupts=1 is needed because GCE
# VMs lack interrupt remapping.
set -uo pipefail

MODE="${1:-start}"
TOKEN="${2:-}"
# ---- EDIT THESE FOR YOUR ENVIRONMENT (or export them before calling) --------
# IMG    : tpu-inference image (vllm + jax + ray + libtpu + editable tpu_inference).
#          Ours needs registry access; if you build your own, the OVERLAY below
#          swaps in the GLM-5.2 fork's tpu_inference package (see CODE_CHANGES.md).
# VBAR   : v6e slice-coordination control agent (public GCR image).
# HEAD_IP: INTERNAL IP of worker 0 (the Ray head). From
#          `gcloud compute tpus tpu-vm describe <VM> --zone .. | grep ipAddress`.
# OVERLAY: the fork's tpu_inference package on the SHARED FS, bind-mounted RO over
#          the in-image package. SHARED_FS/WEIGHTS_FS must be identical on every host.
IMG="${IMG:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference/vllm-tpu:latest}"
VBAR="${VBAR:-gcr.io/cloud-tpu-v2-images/vbar_control_agent:latest}"
HEAD_IP="${HEAD_IP:-10.128.0.101}"
OVERLAY="${OVERLAY:-/mnt/glm5fs/overlay/tpu_inference}"
SHARED_FS="${SHARED_FS:-/mnt/glm5fs}"   # shared writable FS (same path all hosts)
WEIGHTS_FS="${WEIGHTS_FS:-/mnt/glm5}"   # RO weights volume (same path all hosts)
# ----------------------------------------------------------------------------

WNUM=$(curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number)

if [ "$MODE" = "pull" ]; then
  echo "$TOKEN" | sudo docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev >/dev/null 2>&1 && echo "login_uc w=$WNUM"
  echo "$TOKEN" | sudo docker login -u oauth2accesstoken --password-stdin https://gcr.io >/dev/null 2>&1 && echo "login_gcr w=$WNUM"
  sudo docker pull "$IMG"  >/tmp/imgpull.log  2>&1 && echo "PULL_IMG_OK w=$WNUM"  || { echo "PULL_IMG_FAIL w=$WNUM"; tail -3 /tmp/imgpull.log; }
  sudo docker pull "$VBAR" >/tmp/vbarpull.log 2>&1 && echo "PULL_VBAR_OK w=$WNUM" || { echo "PULL_VBAR_FAIL w=$WNUM"; tail -3 /tmp/vbarpull.log; }
  exit 0
fi

# MODE=start
sudo modprobe vfio-pci 2>/dev/null || true
sudo modprobe vfio_iommu_type1 2>/dev/null || true
echo 1 | sudo tee /sys/module/vfio_iommu_type1/parameters/allow_unsafe_interrupts >/dev/null 2>&1 || true
# BIND chips to vfio-pci (libtpu detects/computes via vfio; vbar still reads
# resource4 fine). Bind BEFORE docker run so the container /dev sees the groups.
for d in 0000:00:04.0 0000:00:05.0 0000:00:06.0 0000:00:07.0; do
  cur=$(basename "$(readlink /sys/bus/pci/devices/$d/driver 2>/dev/null)" 2>/dev/null)
  if [ "$cur" != "vfio-pci" ]; then
    echo vfio-pci | sudo tee /sys/bus/pci/devices/$d/driver_override >/dev/null
    echo "$d" | sudo tee /sys/bus/pci/drivers/vfio-pci/bind >/dev/null 2>&1 || true
  fi
done

# (Re)start the vbar control agent (persistent host daemon on :8353).
sudo docker rm -f vbarcontrolagent >/dev/null 2>&1 || true
sudo docker run -d --name=vbarcontrolagent --pid=host --privileged \
  -v /var/run/docker.sock:/var/run/docker.sock -v /tmp:/tmp -v /var/log/:/var/log/ \
  --net=host "$VBAR" \
  vbar_control_agent_files/bin/vbar_control_agent --logtostderr --gid= --uid= --chroot= --census_enabled=false \
  >/dev/null 2>&1 && echo "VBAR_STARTED w=$WNUM" || echo "VBAR_FAIL w=$WNUM"
sleep 6
VBAR_UP=$(sudo ss -tlnp 2>/dev/null | grep -c ":8353 ")

# (Re)start the Ray/vLLM node container.
sudo docker rm -f node >/dev/null 2>&1 || true
RES="--resources='{\"TPU\":4}'"
if [ "$WNUM" = "0" ]; then
  RAY="ray start --head --port=6379 --dashboard-host=0.0.0.0 $RES --block"; ROLE="head"
else
  RAY="ray start --address=${HEAD_IP}:6379 $RES --block"; ROLE="worker"
fi

sudo docker run -d --privileged --network host --shm-size=32G --name node \
  -v "${SHARED_FS}:${SHARED_FS}" \
  -v "${WEIGHTS_FS}:${WEIGHTS_FS}" \
  -v /var/run/tpu-plugin:/var/run/tpu-plugin \
  -v "${OVERLAY}:/workspace/tpu_inference/tpu_inference:ro" \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS= \
  -e RAGGED_GATHER_VERSION=v1 \
  -e RAGGED_GATHER_REDUCE_VERSION=v1 \
  -e MLA_XPOSE_N_TILE_SIZE=64 \
  -e PYTHONPATH=/workspace/vllm:/workspace/tpu_inference \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint /bin/bash \
  "$IMG" -c "$RAY" >/dev/null 2>&1 && echo "NODE_STARTED w=$WNUM role=$ROLE vbar8353=$VBAR_UP" || echo "NODE_FAIL w=$WNUM"
