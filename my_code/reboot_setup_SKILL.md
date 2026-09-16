---
name: heliosr-config
description: Guided, administrator-approved Helios R host configuration runbook for MI450 after the host OS is fully booted and SSH is available. Covers driver load choices by secure-part and DKMS/BKC version, ROCm/HSA setup, BKC-specific performance uplift, and optional AFM validation. Ask the administrator at every branch and require verification after every stage. Never auto-run.
disable-model-invocation: true
---

# Helios R host configuration

This is an interactive **host-only** runbook, not an unattended automation skill.

> Start only after the administrator confirms the OS has fully booted and host
> SSH is available. Host power-control operations are out of scope.
> Never run a command until the administrator explicitly confirms the next
> stage. Do not infer BKC, secure-part status, topology, AFM need, slot ID, or
> ROCm path from a hostname. Ask and record each choice.

## Global execution protocol

For every stage:

1. show the commands that will be run;
2. ask the administrator to approve that stage;
3. run only the approved commands on the explicitly named host (or named BMC only for a separately approved BMC configuration step);
4. show raw output and state the pass/fail condition; and
5. ask the administrator whether to continue, retry, stop, or choose another
   branch.

No machine operation is authorized merely because this skill was selected.

## Decision sheet — collect before starting

Ask for these values one at a time and do not continue with a missing answer:

| decision | required choices |
|---|---|
| target | system hostname and confirmation that host OS boot and SSH are ready |
| part type | unsecure, secure, or unknown; secure parts currently listed: a08-1, c3-1, c3-2 |
| BKC | exact BKC version, plus administrator-selected branch if it sits on a boundary |
| topology | 1P1G, 1P2G, or 1P4G; delegate 1P4G special configuration to heliosr-1p4g-config |
| AFM | no AFM, inband AFM, or a different administrator-specified access method |
| validation | host-only, AFM/UALink, ubench06, and/or TDM validation |

Do not store BMC, AMC, Docker, or other credentials in the skill, terminal
history, or workspace. Request them through the administrator's approved secret
path only when a command genuinely requires them.

# Stage 1 — host driver and ROCm setup

Confirm host SSH is stable before continuing.

## Driver and ROCm setup

## 2A. Record driver and firmware state

~~~bash
dkms status
sudo cat /sys/kernel/debug/dri/1/amdgpu_firmware_info
~~~

Report the complete DKMS line and firmware output. Extract the numeric driver
build (for example, 7.1.1-2389086.el9 has build 2389086).

**Decision gate:** if the build is older than 2351230, stop. Ask the
administrator to provide or approve the corresponding command from:
https://amd.atlassian.net/wiki/spaces/AMDGPU/pages/1498351269

Do not guess an older-driver modprobe command.

## 2B. Select the driver-load path

Ask the administrator to choose one:

1. **BKC26.09.xx onward** — direct to Stage 3; do not load the driver here.
2. **Unsecure part** — after approval:

~~~bash
sudo modprobe amdgpu gpu_recovery=0 halt_if_hws_hang=1
~~~

3. **Secure part** (a08-1, c3-1, c3-2) — after approval:

~~~bash
sudo modprobe amdgpu gpu_recovery=0 ip_block_mask=0xcff
~~~

**Verification gate:** after the selected command, ask to perform exactly one
ROCm enumeration:

~~~bash
rocm-smi || /opt/rocm/bin/rocm-smi
~~~

Report the enumerated GPU count. Do not continue if ROCm fails or count disagrees
with the selected topology. Do not repeat rocm-smi without an administrator
request; it creates additional GPU-context activity.

## 2C. Set ROCm paths and persist runtime environment

Show this plan and ask approval:

~~~bash
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode

export HSA_ENABLE_SDMA=1
export HSA_USE_SVM=1
export HSA_XNACK=1
export LOADER_ENABLE_TRAMPOLINE=1
export AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES=0
~~~

For persistence, ask whether this should be user-only or system-wide. Do not
choose silently. For system-wide persistence, write an administrator-reviewed
profile script under /etc/profile.d and verify its exact contents. For user-only
persistence, write to the approved user shell profile only after confirmation.

**Verification gate:** open a fresh shell or source the approved profile, then
print the seven variables and verify their values.

# Stage 3 — topology record

Record the administrator-confirmed topology (1P1G, 1P2G, or 1P4G) for
validation and BKC branch selection only. **Do not configure agt_internal and
do not send AGT SMC messages.** The former 0x3D,0x10032 tuning commands are
intentionally out of scope.


# Stage 4 — BKC-specific platform performance uplift

Ask the administrator to select the exact BKC branch. Do not resolve ambiguous
version boundaries yourself.

## Branch A: BKC26.09.xx onward

~~~bash
wget http://dcgpuval-storage.amd.com/users/cheechia/MI450/agt/perf_setup_BPC19.sh
sudo chmod +x perf_setup_BPC19.sh
sudo ./perf_setup_BPC19.sh -enable-csc
# At its prompt, choose: NC load amdgpu
sudo ddvs regmi w GFX_ICG_TCP_CTRL2 'xcd*' 0x2
sudo cpupower frequency-set -g performance
~~~

**Verification gate:** show the interactive script transcript, ddvs result, and
CPU governor. Skip the legacy curl-script branches after this branch, then go to
the AFM decision.

## Branch B: >= BKC26.06.03 and < BKC26.09.01

Ask approval for the documented legacy sequence:

~~~bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
curl -fsSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/set_dram_settings.py | sudo python3
curl -fsSL http://dcgpuval-storage.amd.com/users/kstraube/set_tdc_limits_mi450.py | sudo python3
curl -fsSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/disable_gcea_link_mgr.py | sudo python3
curl -fsSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/set_cp_hpd_enable_offload_check.py | sudo python3
curl -fsSL http://dcgpuval-storage.amd.com/users/muku/MI450x_ScaleUp_PerfScripts/MI450_disTxIdle_may13.py | sudo python3
curl -fsSL http://dcgpuval-storage.amd.com/users/harnsing_pharaoh/kll_optimization/kll_optimization_mi450.py | sudo python3
curl -fsSL http://dcgpuval-storage.amd.com/users/tifyeung/debug_scripts/MI450_DF_DisSDPdis.py | sudo python3
curl -fsSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/disable_mgcg_override.py | sudo python3
curl -fsSL http://dcgpuval-storage.amd.com/users/abalam/MI45X/scripts/df_mi450_sec_lvl.py | sudo python3
curl -sSL http://dcgpuval-storage.amd.com/users/muku/MI450x_ScaleUp_PerfScripts/ifoe_moderationTO.py | sudo python3 - 20
curl -sSL http://dcgpuval-storage.amd.com/users/tifyeung/debug_scripts/df_ucakecomp_dis.py | sudo python3
~~~

## Branch C: < BKC26.06.03

Use Branch B unchanged except replace the MGCG command with:

~~~bash
curl -fsSL http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/mgcg_override.py | sudo python3
~~~

**Verification gate for B/C:** run and report:

~~~bash
cat /proc/sys/kernel/numa_balancing
sudo dmesg | grep -Eia 'MES.*remove.?queue' || true
~~~

A MES remove-queue signature is a stop condition. Do not declare the machine
healthy; report it to the monitor/administrator.

# Stage 5 — optional AFM and validation

Ask whether AFM is required. If no, stop and report completed stages. If yes,
ask whether the intended access method is **inband AFM**. The documented agent
commands below are for inband AFM only.

~~~bash
sudo modprobe ifoe
sudo modprobe ifoe_cfg
sudo modprobe ifoe_cmd
lsmod | grep ifoe
~~~

Ask the administrator to choose an agent source:

1. existing /opt/amd/aifm_agent_pkg:

~~~bash
cd /opt/amd/aifm_agent_pkg
./start_aifm_agent.sh
~~~

2. general package path (requires approved package source and slot ID):

~~~bash
# obtain package through approved internal source
# unpack it, then:
export LD_LIBRARY_PATH=/opt/rocm/lib/
./afm_agent.sh start node SLOT_ID
~~~

**Verification gate:**

~~~bash
sudo afmctl show device
~~~

Wait for configuration phase ACTIVE. Then ask whether to check the original or
new UALink sysfs layout, and run only the selected layout:

~~~bash
# original layout
sudo cat /sys/class/drm/card*/device/ualink/local_accels
sudo cat /sys/class/drm/card*/device/ualink/setup/local_accels
sudo cat /sys/class/drm/card*/device/ualink/stations/lane_en_bitmap
sudo cat /sys/class/drm/card*/device/ualink/accel_state

# newer layout
sudo cat /sys/class/drm/card*/device/ualink/local_accels
sudo cat /sys/class/drm/card*/device/ualink/station_lane_en_bitmap
sudo cat /sys/class/drm/card*/device/ualink/accel_state
~~~

For ubench06 or TDM validation, ask for the approved benchmark path/image,
ROCm-path exception, expected result, and explicit execution authorization.
Never embed registry credentials or passwords in this skill.
