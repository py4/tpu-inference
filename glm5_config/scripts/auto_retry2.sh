#!/bin/bash
# Auto-retry the full GLM-5.2 load+generate across pod restarts (the pathways
# IFRT proxy intermittently resets during the 1.4TB host->TPU transfer; 2/4 runs
# reached 99%, so retrying may eventually complete). Runs on the LOGIN node so it
# survives pod restarts. Re-overlays the fork + relaunches whenever no run is
# active and there's no success yet. Stops on GEN_OK.
L=/home/pooyam_google_com/projects/glm_5.2
LOG=/home/pooyam_google_com/auto_retry.log
echo "=== auto-retry started $(date) ===" >> $LOG
attempt=0
while true; do
  POD=$(kubectl get pods -o name 2>/dev/null | grep pathways-head | head -1 | sed 's,pod/,,')
  if [ -z "$POD" ]; then echo "$(date) no pod yet" >> $LOG; sleep 60; continue; fi
  READY=$(kubectl get pod "$POD" 2>/dev/null | awk 'NR==2{print $2,$3}')
  if ! echo "$READY" | grep -qE "3/3 Running"; then echo "$(date) pod $POD $READY (waiting)" >> $LOG; sleep 60; continue; fi
  # alive check
  if [ "$(kubectl exec $POD -c main -- echo ok 2>/dev/null)" != "ok" ]; then sleep 30; continue; fi
  # success?
  if kubectl exec $POD -c main -- bash -c 'grep -q GEN_OK /mnt/ckpt/glm5_gen.log 2>/dev/null' 2>/dev/null; then
    echo "$(date) GEN_OK!! SUCCESS on $POD" >> $LOG; break
  fi
  # running?
  N=$(kubectl exec $POD -c main -- pgrep -fc glm5_generate 2>/dev/null | tr -d '[:space:]')
  if [ "$N" -ge 1 ] 2>/dev/null; then echo "$(date) gen running on $POD ($N procs)" >> $LOG; sleep 300; continue; fi
  # not running, no success -> re-overlay + relaunch
  attempt=$((attempt+1))
  echo "$(date) === relaunch attempt $attempt on $POD ===" >> $LOG
  kubectl exec $POD -c main -- bash -c '[ -d /deps/src/tpu_inference ] && mv /deps/src/tpu_inference /deps/src/tpu_inference.disabled 2>/dev/null; rm -rf /workspace/tpu_inference/tpu_inference; mkdir -p /workspace/glm5_config' >>$LOG 2>&1
  kubectl cp $L/tpu-inference/tpu_inference $POD:/workspace/tpu_inference/ -c main >>$LOG 2>&1
  kubectl cp $L/glm5_config $POD:/workspace/glm5_config_tmp -c main >>$LOG 2>&1
  kubectl exec $POD -c main -- bash -c 'cp /workspace/glm5_config_tmp/* /workspace/glm5_config/ 2>/dev/null' >>$LOG 2>&1
  timeout 30 kubectl exec $POD -c main -- bash -c 'cd /workspace && setsid env JAX_PLATFORMS=proxy,cpu NEW_MODEL_DESIGN=1 VLLM_TPU_USING_PATHWAYS=1 GLM_NUM_EXPERTS=256 SMOKE_TP=64 SMOKE_EP=64 SKIP_JAX_PRECOMPILE=1 MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn PYTHONUNBUFFERED=1 GEN_MODEL=/mnt/ckpt/glm5 GEN_MAXTOK=48 python3 /workspace/glm5_config/glm5_generate.py >/mnt/ckpt/glm5_gen.log 2>&1 </dev/null & echo launched' >>$LOG 2>&1
  echo "$(date) launched attempt $attempt" >> $LOG
  sleep 300
done
