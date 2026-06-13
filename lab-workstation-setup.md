# Research Lab Workstation Setup Guide

A practical guide for configuring a shared multi-user GPU workstation for an ML
research lab (3–4 simultaneous users), based on a real setup. Reorganized into a
logical sequence rather than the order issues were discovered.

## Table of Contents

- [0. Hardware & OS](#0-hardware--os)
- [1. Multi-user foundation](#1-multi-user-foundation)
  - [Separate accounts](#separate-accounts-never-share-one-login)
  - [SSH access](#ssh-access-the-backbone-for-34-simultaneous-users)
  - [Shared directory](#shared-directory-for-collaboration)
  - [Rename the machine](#rename-the-machine-optional)
- [2. Resource limits](#2-resource-limits-so-one-user-cant-starve-the-box)
  - [CPU / memory per user](#cpu--memory-per-user-systemd-slices)
  - [Disk quotas](#disk-quotas-4-tb-drive)
- [3. NVIDIA driver](#3-nvidia-driver)
- [4. MIG (Multi-Instance GPU)](#4-mig-multi-instance-gpu--the-gpu-sharing-strategy)
  - [The hybrid layout](#the-hybrid-layout-recommended-for-a-shared-lab)
  - [Prerequisites](#prerequisites-all-required)
  - [Switch to compute mode](#switch-the-target-card-to-compute-mode)
  - [Enable MIG and create instances](#enable-mig-and-create-instances)
  - [Persist MIG instances across reboots](#persist-mig-instances-across-reboots)
  - [Troubleshooting: instances missing after reboot](#troubleshooting-mig-instances-missing-after-reboot)
  - [Using MIG in PyTorch](#using-mig-in-pytorch)
- [5. Shared software](#5-shared-software)
- [6. Shared conda + per-user environments](#6-shared-conda--per-user-environments)
- [7. Remote access, persistent jobs & file transfer](#7-remote-access-persistent-jobs--file-transfer)
  - [Persistent jobs (tmux)](#keep-jobs-alive-after-disconnect-no-slurm-needed)
  - [Worked example: training in tmux](#worked-example-a-logged-training-run-in-tmux)
  - [File transfer](#file-transfer)
  - [GUI for users](#gui-for-users)
- [8. Monitoring](#8-monitoring-real-time)
- [Outstanding to-dos](#outstanding-to-dos)

---

## 0. Hardware & OS

- **Workstation:** 2× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition,
  96 GB each (192 GB total GPU memory), 4 TB NVMe, Ubuntu 24.04 LTS.
- **GPU compute capability:** sm_120 (Blackwell).

---

## 1. Multi-user foundation

### Create and config new account (never share one login)

```bash
admin@wenlab-workstation:~$ sudo adduser alice
admin@wenlab-workstation:~$ sudo usermod -aG sudo alice    # only for admins who need to install software
```

### Enable SSH access (the backbone for 3–4 simultaneous users)

```bash
admin@wenlab-workstation:~$ sudo apt update
admin@wenlab-workstation:~$ sudo apt install openssh-server
admin@wenlab-workstation:~$ sudo systemctl enable --now ssh
```

Find the workstation's IP to connect to:

```bash
admin@wenlab-workstation:~$ hostname -I            # first address is usually the one to use
admin@wenlab-workstation:~$ ip route get 1.1.1.1   # the "src" value is the active IP
```

Prefer a **static IP or DHCP reservation** so the address doesn't change on
reboot. Users connect with `ssh alice@<ip>`. This works directly when you're on
the **same network** as the workstation (e.g. the lab or campus network/Wi-Fi);
from off-site you'd need a VPN into that network. From their own machine users
open a terminal (Linux/Mac/WSL), PowerShell (Windows with OpenSSH), or an SSH
client like PuTTY/MobaXterm, then enter their password when prompted.

### Setup Shared directory for collaboration

```bash
admin@wenlab-workstation:~$ sudo mkdir /srv/shared
admin@wenlab-workstation:~$ sudo groupadd labusers
admin@wenlab-workstation:~$ sudo usermod -aG labusers alice    # add each user
admin@wenlab-workstation:~$ sudo chown root:labusers /srv/shared
admin@wenlab-workstation:~$ sudo chmod 2775 /srv/shared        # setgid: new files inherit the group
```

### Rename the machine (optional)

```bash
admin@wenlab-workstation:~$ sudo hostnamectl set-hostname wenlab-workstation
admin@wenlab-workstation:~$ sudo nano /etc/hosts               # update the 127.0.1.1 line to match
```

Use lowercase letters, digits, hyphens only. Reachable at
`wenlab-workstation.local` via mDNS.

### Access the SSH on Remote PC

Once SSH is enabled on the workstation, users connect from their own machine
using their account name and the workstation's IP (or `wenlab-workstation.local`
if you're on the same network and mDNS works). Replace `alice` with your
username and `<ip>` with the address from `hostname -I`.

**1. Linux (Ubuntu / Fedora / Mint, etc.)** — Open a terminal and run:

```bash
alice@laptop:~$ ssh alice@<ip>                      # e.g. ssh alice@192.168.1.42
alice@laptop:~$ ssh alice@wenlab-workstation.local  # if on the same network
```

The first time you connect you'll be asked to confirm the host's fingerprint —
type `yes`, then enter your password when prompted.

> **Troubleshoot — `ssh: command not found`?** The OpenSSH client isn't
> installed. Install it with your distro's package manager:
>
> ```bash
> alice@laptop:~$ sudo apt install openssh-client      # Debian / Ubuntu / Mint
> alice@laptop:~$ sudo dnf install openssh-clients     # Fedora / RHEL
> alice@laptop:~$ sudo pacman -S openssh               # Arch
> ```

**2. macOS** — The built-in Terminal (or iTerm2) ships with an SSH client, so the
command is identical to Linux:

```bash
alice@macbook:~$ ssh alice@<ip>
```

> **Troubleshoot — `ssh: command not found`?** OpenSSH ships with macOS, so this
> is rare. If it's missing, install the Command Line Tools (which include `ssh`)
> or use [Homebrew](https://brew.sh/):
>
> ```bash
> alice@macbook:~$ xcode-select --install   # installs Apple's command line tools
> alice@macbook:~$ brew install openssh     # or via Homebrew
> ```

**3. Windows** — There are two common options:

- **PowerShell / Command Prompt (built-in OpenSSH client):** modern Windows 10/11
  includes `ssh`, so you can connect directly:

  ```powershell
  PS C:\Users\alice> ssh alice@<ip>
  ```

  > **Troubleshoot — `ssh : The term 'ssh' is not recognized`?** The OpenSSH
  > client isn't installed. Add it via *Settings → Apps → Optional features →
  > Add a feature → OpenSSH Client*, or run this in an **Administrator**
  > PowerShell:
  >
  > ```powershell
  > PS C:\Users\alice> Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
  > ```
  >
  > Close and reopen the terminal afterward so `ssh` is on your `PATH`.

- **PuTTY / MobaXterm (GUI clients):** download [PuTTY](https://www.putty.org/)
  or [MobaXterm](https://mobaxterm.mobatek.net/), enter the workstation's IP in
  the **Host Name** field, keep **Port 22** and **SSH** selected, click **Open**,
  then log in with your username and password.

> **Tip — passwordless login (all platforms):** generate a key pair once with
> `ssh-keygen -t ed25519`, copy the public key to the workstation with
> `ssh-copy-id alice@<ip>` (on Windows, use
> `type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh alice@<ip> "cat >> ~/.ssh/authorized_keys"`),
> and you'll no longer be prompted for a password on future connections.

---

## 2. Resource limits (so one user can't starve the box)

### CPU / memory per user (systemd slices)

```bash
# UID 1001 example: cap at ~200% CPU (2 cores) and 32 GB RAM
admin@wenlab-workstation:~$ sudo systemctl set-property user-1001.slice CPUQuota=200% MemoryMax=32G
```

To revert (empty = reset to unlimited):

```bash
admin@wenlab-workstation:~$ sudo systemctl set-property user-1001.slice CPUQuota="" MemoryMax=""
```

### Disk quotas (4 TB drive)

> **Hard-won lesson:** a typo in `/etc/fstab` (`defaut` instead of `defaults`)
> caused a boot failure. **Always validate fstab before rebooting.**

1. Enable quotas on the root filesystem:

   ```bash
   admin@wenlab-workstation:~$ sudo apt install quota
   admin@wenlab-workstation:~$ sudo nano /etc/fstab
   ```

   Add `usrquota` to the root line's options:

   ```
   /dev/disk/by-uuid/<uuid> / ext4 defaults,usrquota 0 1
   ```

2. **Validate before rebooting** (the safety step):

   ```bash
   admin@wenlab-workstation:~$ sudo findmnt --verify --verbose
   admin@wenlab-workstation:~$ sudo mount -o remount /         # fails safely if there's a typo
   ```

3. Initialize and turn on:

   ```bash
   admin@wenlab-workstation:~$ sudo quotacheck -cvum /
   admin@wenlab-workstation:~$ sudo quotaon -v /
   ```

4. Set a per-user limit (block numbers are in 1 KB blocks;
   GB × 1024 × 1024 = blocks). Example: user `alice`, 512 GB soft / 800 GB hard:

   ```bash
   admin@wenlab-workstation:~$ sudo setquota -u alice 536870912 838860800 0 0 /
   admin@wenlab-workstation:~$ sudo repquota -s /              # verify, human-readable
   ```

> **Recovery note:** if a bad fstab ever blocks boot, get a console
> (Ctrl+Alt+F3 or GRUB → recovery → root shell), then remount read-write
> *bypassing fstab* by giving both device and mountpoint:
> `sudo mount -o remount,rw /dev/<device> /`, fix the file, reboot.

---

## 3. NVIDIA driver

Blackwell cards require recent, **open-kernel-module** drivers.

```bash
admin@wenlab-workstation:~$ ubuntu-drivers list
admin@wenlab-workstation:~$ sudo apt install nvidia-driver-595-open    # newest -open, NOT -server for a workstation w/ display
admin@wenlab-workstation:~$ sudo reboot
admin@wenlab-workstation:~$ nvidia-smi                                 # confirm Driver Version 595.xx, CUDA 13.x
```

- Use **`-open`** (required for Blackwell).
- Use the **non-`-server`** desktop variant on a machine with a display.
- Minimum for MIG on these cards is R575; 595 is fine.
- Removing old drivers may break pre-installed system CUDA/PyTorch packages
  (`apt --fix-broken install` to clean up — fine to remove them since we use
  per-user conda).

---

## 4. MIG (Multi-Instance GPU) — the GPU-sharing strategy

### The hybrid layout (recommended for a shared lab)

- **GPU 0:** left in normal mode → a whole 96 GB card; also drives the display;
  used for big single jobs needing the full card.
- **GPU 1:** partitioned with MIG into isolated slices (e.g. four × 24 GB) →
  one dedicated slice per user, hardware-isolated (no OOM-ing each other).

MIG memory comes in fixed **24 GB slices** (4 per card), so valid sizes are
multiples of 24: 24, 48, or 96 GB. No 32 GB option. For 3 instances you choose
between `3×24` (one slice idle) or `48+24+24`.

### Prerequisites (all required)

1. Driver ≥ R575 (done above).
2. vBIOS meets the minimum (`nvidia-smi --query-gpu=vbios_version --format=csv`;
   Max-Q minimum is `98.02.6A.00.00` — contact reseller if older).
3. Resizable BAR (ReBAR) enabled in system BIOS.
4. **Display mode = compute** on the card to be sliced (next step).

### Switch the target card to compute mode

The Workstation/Max-Q editions ship in **graphics** mode; MIG needs **compute**
mode. Use NVIDIA's **Display Mode Selector** tool (gated download — get it via a
free NVIDIA Developer login or your reseller, then `scp` it over). Do everything
**over SSH** and target the card **not** driving your display.

```bash
admin@wenlab-workstation:~$ unzip NVIDIA-Display-Mode-Selector-Tool-*.zip
admin@wenlab-workstation:~$ cd "NVIDIA-Display-Mode-Selector-Tool-*/linux/x64/"
admin@wenlab-workstation:~$ chmod +x displaymodeselector

admin@wenlab-workstation:~$ sudo ./displaymodeselector --list          # map adapters to bus IDs
admin@wenlab-workstation:~$ sudo ./displaymodeselector --listgpumodes  # check current modes

# unload the driver first (tool refuses to run while it's loaded):
admin@wenlab-workstation:~$ sudo systemctl isolate multi-user.target
admin@wenlab-workstation:~$ sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia

```

Now run the compute switch **interactively** — this is the critical step:

```bash
admin@wenlab-workstation:~$ sudo ./displaymodeselector --gpumode compute
```

Answer the prompts like this:

```
WARNING: This operation updates the firmware on the board ...
Press 'y' to confirm (any other key to abort):
y                              # confirm

Update GPU Mode of all adapters to "physical_display_disabled"?
Press 'y' to confirm or 'n' to choose adapters or any other key to abort:
n                              # press 'n' (NOT 'y') so you can choose ONE card

Select a display adapter:
<0> Graphics Device  ... B:C1 ...   <- GPU 0, the display card: leave it
<1> Graphics Device  ... B:E1 ...   <- GPU 1, the card to slice
Select a number (ESC to quit):
1                              # choose GPU 1 (bus E1)
```

Then reboot and confirm the switch:

```bash
admin@wenlab-workstation:~$ sudo reboot
# after reboot:
admin@wenlab-workstation:~$ sudo ./displaymodeselector --listgpumodes   # adapter 1 should now read Compute
```

> **Critical:** at the "all adapters?" prompt press **`n`**, then select the
> single non-display card. Pressing `y` converts *both* cards and blanks your
> local display entirely. Match the adapter index to `nvidia-smi` by PCI **bus
> ID** (bus E1 = GPU 1). If GPUs don't appear in compute mode afterward, some
> report needing the `-server-open` datacenter driver variant.

### Enable MIG and create instances

```bash
admin@wenlab-workstation:~$ sudo nvidia-smi -i 1 -mig 1
admin@wenlab-workstation:~$ nvidia-smi -i 1 --query-gpu=mig.mode.current --format=csv   # expect "Enabled"
admin@wenlab-workstation:~$ sudo nvidia-smi mig -i 1 -lgip                              # list profiles + IDs
```

Profile IDs on this card: `14` = 1g.24gb, `5` = 2g.48gb, `0` = 4g.96gb.

```bash
admin@wenlab-workstation:~$ sudo nvidia-smi mig -i 1 -cgi 14,14,14,14 -C   # four 24GB instances
admin@wenlab-workstation:~$ nvidia-smi -L                                  # shows MIG devices + UUIDs
```

> **"In use by another client" error:** the desktop (Xorg, gnome-shell, mutter,
> portals) holds a render handle on the GPU. Stop the whole session as a unit —
> don't kill processes individually:
> ```bash
> admin@wenlab-workstation:~$ sudo systemctl isolate multi-user.target
> admin@wenlab-workstation:~$ sudo fuser -v /dev/nvidia1        # confirm it's clear
> admin@wenlab-workstation:~$ sudo nvidia-smi mig -i 1 -cgi 14,14,14,14 -C
> admin@wenlab-workstation:~$ sudo systemctl isolate graphical.target
> ```
> Once instances exist, the desktop can't grab that card again, so this is a
> one-time setup conflict.

### Persist MIG instances across reboots

MIG **mode** survives a reboot, but the **instances do not** — they're wiped on
every boot and must be recreated. A small systemd service handles this, ordered
to run **before the display manager** so GPU 1 is free when instances are
created (avoiding the "in use" error above).

Create the recreation script `/usr/local/sbin/setup-mig.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail

GPU=1                    # the MIG-partitioned card
PROFILES="14,14,14,14"   # four 1g.24gb slices (use "5,14,14" for 48+24+24, etc.)
WANT=4                   # expected number of MIG devices

for i in $(seq 1 30); do                       # wait for the driver at early boot
    nvidia-smi -i "$GPU" >/dev/null 2>&1 && break
    sleep 1
done

if [ "$(nvidia-smi -L | grep -c 'MIG ')" -ge "$WANT" ]; then
    echo "MIG instances already present; nothing to do."; exit 0
fi

nvidia-smi -i "$GPU" -mig 1 || true            # ensure MIG mode is on
nvidia-smi mig -i "$GPU" -dci 2>/dev/null || true
nvidia-smi mig -i "$GPU" -dgi 2>/dev/null || true
nvidia-smi mig -i "$GPU" -cgi "$PROFILES" -C   # recreate the layout
```

Create the service `/etc/systemd/system/mig-setup.service`:

```ini
[Unit]
Description=Recreate NVIDIA MIG instances on GPU 1 at boot
After=local-fs.target
Before=display-manager.service graphical.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/setup-mig.sh

[Install]
WantedBy=multi-user.target
```

Make it executable, enable the service, and test (safe — idempotent):

```bash
admin@wenlab-workstation:~$ sudo chmod +x /usr/local/sbin/setup-mig.sh
admin@wenlab-workstation:~$ sudo systemctl daemon-reload
admin@wenlab-workstation:~$ sudo systemctl enable mig-setup.service
admin@wenlab-workstation:~$ sudo systemctl start mig-setup.service
admin@wenlab-workstation:~$ systemctl status mig-setup.service --no-pager
```

Then reboot and confirm the instances return:

```bash
admin@wenlab-workstation:~$ sudo reboot
# after reboot:
admin@wenlab-workstation:~$ nvidia-smi -L         # four MIG 1g.24gb devices again
```

The script is **idempotent** (exits early if the instances already exist), so
running it live won't disturb a working setup. To change the layout later, edit
only `PROFILES` and `WANT`. If the service ever shows `failed`, check
`journalctl -u mig-setup.service` — usually the GPU wasn't ready within 30 s, so
raise the loop count.

### Troubleshooting: MIG instances missing after reboot

If `nvidia-smi -L` shows no MIG devices after a reboot, or `mig-setup.service`
reports `failed`, work through these.

**First, read the service logs — this needs `sudo`:**

```bash
admin@wenlab-workstation:~$ sudo systemctl status mig-setup.service --no-pager
admin@wenlab-workstation:~$ sudo journalctl -u mig-setup.service -b
```

(Without `sudo`, journalctl prints "No entries" unless your user is in the `adm`
or `systemd-journal` group. The `-b` flag limits output to the current boot.)

**Exit status 19 / "In use by another client".** Status 19 is nvidia-smi's
`NVML_ERROR_IN_USE` — GPU 1 was busy when the script ran. This happens when you
start the service **manually while the desktop is running** (GNOME holds GPU 1),
and it's expected, *not* a real fault: the service is designed to run at boot,
before the display manager. Either reboot (it runs before the desktop, so the
GPU is free), or create the instances now by freeing the GPU first:

```bash
admin@wenlab-workstation:~$ sudo systemctl isolate multi-user.target
admin@wenlab-workstation:~$ sudo systemctl start mig-setup.service
admin@wenlab-workstation:~$ sudo systemctl isolate graphical.target
admin@wenlab-workstation:~$ nvidia-smi -L
```

**It fails at every boot (not just manual starts).** Check this boot's log
(`sudo journalctl -u mig-setup.service -b`) and match the message:

- *"in use" even at boot* → something grabbed GPU 1 before the service ran.
  Confirm the unit has `Before=display-manager.service graphical.target`; if a
  boot splash is the culprit, add `After=plymouth-quit.service` to `[Unit]`.
- *driver/GPU not ready, or a timeout* → the GPU wasn't initialized within the
  30-second wait. Raise the loop count in `/usr/local/sbin/setup-mig.sh`
  (e.g. `seq 1 60`).
- *"command not found" / syntax error* → the script didn't write correctly via
  the heredoc; recreate it and verify with `cat /usr/local/sbin/setup-mig.sh`.

**Confirm the service is actually enabled:**

```bash
admin@wenlab-workstation:~$ systemctl is-enabled mig-setup.service
```

It should print `enabled`. If it says `disabled`, run
`sudo systemctl enable mig-setup.service`.

### Using MIG in PyTorch

A process uses **one** MIG instance, selected outside Python via a MIG **UUID**;
inside the process it's always `cuda:0`.

```bash
admin@wenlab-workstation:~$ nvidia-smi -L                                       # grab a MIG-... UUID
admin@wenlab-workstation:~$ CUDA_VISIBLE_DEVICES=MIG-<uuid> python train.py
```

```python
import torch
device = torch.device("cuda:0")   # the slice you exposed
```

- `cuda:0` is **not** automatically nvidia-smi's GPU 0 — control placement with
  `CUDA_VISIBLE_DEVICES` (by UUID) rather than relying on index order.
- For the whole non-MIG card (GPU 0): expose its `GPU-...` UUID the same way.
- **Blackwell needs a recent PyTorch** (CUDA 12.8+ build / sm_120 kernels), e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- Set each user's assigned MIG UUID in their `~/.bashrc` (or use `genv`) so they
  just run jobs without thinking about UUIDs.

---

## 5. Shared software

Anything installed with `apt` is system-wide automatically:

```bash
admin@wenlab-workstation:~$ sudo apt install build-essential git tmux htop nvtop curl wget
```

- Software not in apt → install to `/opt` or `/usr/local` (on everyone's PATH).
- **MATLAB:** run the installer with `sudo`, install to the default
  `/usr/local/MATLAB/R20XXx` (shared), symlink `matlab` into `/usr/local/bin`.
  Confirm the license is **Network/Concurrent** or **Designated Computer**
  (covers all users), not Individual.

> **Cross-platform code gotcha:** ported Windows code using `\` path separators
> breaks on Linux (e.g. `d1_GL\10_10_GL.mat`). Fix by joining paths with
> `fullfile()` (MATLAB) / `os.path.join` (Python), not literal backslashes.

---

## 6. Shared conda + per-user environments

Install conda once to a shared location (**not** a home directory):

```bash
admin@wenlab-workstation:~$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
admin@wenlab-workstation:~$ sudo bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
# add -u if /opt/conda already exists
```

Make it available to **all** users — both login *and* non-login shells:

```bash
admin@wenlab-workstation:~$ echo '. /opt/conda/etc/profile.d/conda.sh' | sudo tee /etc/profile.d/conda.sh
admin@wenlab-workstation:~$ echo '. /opt/conda/etc/profile.d/conda.sh' | sudo tee -a /etc/bash.bashrc
```

> **Shell gotcha:** `/etc/profile.d/` is read only by *login* shells; GUI
> terminals are *non-login* and read `/etc/bash.bashrc`. You need both lines.

Verify it's globally accessible (the definitive test runs as another user):

```bash
admin@wenlab-workstation:~$ which conda                  # should be /opt/conda/bin/conda, not a home dir
admin@wenlab-workstation:~$ sudo -u alice bash -lc 'conda --version'
```

> If `which conda` points into a home directory, **don't move it** — conda
> hardcodes absolute paths everywhere. Install fresh to `/opt/conda`, remove the
> personal install's `conda initialize` block from that user's `~/.bashrc`, and
> migrate envs with `conda env export` / `conda env create -f`.

**Best practice:** shared read-only base envs in `/opt/conda/envs` for common
stacks; each user creates private envs in their home dir. CUDA comes per-env via
conda/pip, so no system CUDA toolkit needed.

---

## 7. Remote access, persistent jobs & file transfer

### Keep jobs alive after disconnect (no SLURM needed)

Closing your SSH session normally kills whatever you were running. To keep a job
going after you log out (or your Wi-Fi drops, or you close your laptop), run it
inside a **terminal multiplexer** — `tmux` is the standard choice. You do *not*
need SLURM for this; SLURM is a job *scheduler* (queuing across many jobs), which
is overkill just to survive a disconnect.

**tmux — the recommended approach.** `tmux` runs a shell on the workstation that
lives independently of your SSH connection. Start a named session:

```bash
admin@wenlab-workstation:~$ tmux new -s train
```

Inside it, launch your job normally (e.g. `python train.py`). Then **detach** by
pressing `Ctrl+b`, releasing, then `d`. You're back at your normal shell and the
job keeps running on the workstation — you can now safely close SSH.

When you return, SSH back in and reattach:

```bash
admin@wenlab-workstation:~$ tmux ls                  # list your running sessions
admin@wenlab-workstation:~$ tmux attach -t train     # reattach to "train"
```

Other handy commands:

```bash
admin@wenlab-workstation:~$ tmux new -s data             # a second named session
admin@wenlab-workstation:~$ tmux kill-session -t train   # stop a session when done
```

If `tmux ls` prints `no server running on ...`, it just means there are no
sessions right now — nothing is wrong. Sessions are per machine, and each user
sees only their own.

**nohup — for fire-and-forget jobs.** If you only want to launch something and
not watch it interactively, `nohup` detaches a single command and logs its
output to a file:

```bash
admin@wenlab-workstation:~$ nohup python train.py > train.log 2>&1 &
```

The `&` backgrounds it, `nohup` makes it ignore the disconnect (hangup) signal,
and you check progress in `train.log`. Simpler than tmux, but you can't reattach
to a live view — good for batch scripts, less ideal for monitoring a long run.

**Already started a job in the foreground?** If you only realise mid-run that you
need to disconnect, press `Ctrl+z` to suspend it, then:

```bash
admin@wenlab-workstation:~$ bg        # resume it in the background
admin@wenlab-workstation:~$ disown    # detach it from the shell so it survives logout
```

**Which to use:** `tmux` for anything you want to monitor, reattach to, or run
interactively (most ML training — and you can keep `htop`/`nvitop` in other tmux
panes while training). `nohup` for simple batch jobs you'll check via a log file.
SLURM only if you outgrow manual coordination and need a real submission queue.

### Worked example: a logged training run in tmux

A complete end-to-end example tying MIG, logging, and tmux together: a small
training script that **logs to a file**, run on a MIG slice **inside tmux**, so
you can disconnect and later confirm it kept running. Save this as `train_demo.py`:

```python
import logging, os, sys, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# log to BOTH a file (to check after you disconnect) and the console (tmux pane)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("train.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger()

log.info("RUN START  pid=%d  CUDA_VISIBLE_DEVICES=%s",
         os.getpid(), os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))

device = torch.device("cuda:0")           # the single MIG slice you exposed
log.info("device: %s  (%.1f GiB)", torch.cuda.get_device_name(0),
         torch.cuda.get_device_properties(0).total_memory / 1024**3)

tf = transforms.Compose([transforms.ToTensor()])
loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=tf),
                    batch_size=256, shuffle=True, num_workers=2)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                      nn.Linear(256, 10)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(20):                    # long enough to detach and reconnect
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        if step % 100 == 0:
            log.info("epoch %d  step %d  loss %.4f", epoch, step, loss.item())
log.info("RUN COMPLETE")
```

Now run it in tmux. Find a MIG UUID, start a session, and launch the job pinned
to that slice:

```bash
admin@wenlab-workstation:~$ nvidia-smi -L                # copy a MIG-... UUID
admin@wenlab-workstation:~$ tmux new -s train
# --- you are now inside the tmux session ---
admin@wenlab-workstation:~$ CUDA_VISIBLE_DEVICES=MIG-9f6a1ac8-... python train_demo.py
```

Detach with `Ctrl+b` then `d`, then close SSH. Later, reconnect and watch the
log — the timestamps prove it kept training while you were gone:

```bash
admin@wenlab-workstation:~$ tail -f train.log
```

```
14:02:11  RUN START  pid=8661  CUDA_VISIBLE_DEVICES=MIG-9f6a1ac8-...
14:02:13  device: NVIDIA RTX PRO 6000 Blackwell Max-Q ... MIG 1g.24gb  (23.6 GiB)
14:02:15  epoch 0  step 0  loss 2.3041
14:02:41  epoch 1  step 0  loss 0.3922
...
14:11:50  RUN COMPLETE
```

Or jump back into the live session with `tmux attach -t train`.

Two checks are built in: timestamps spanning the time you were disconnected
confirm tmux kept the job alive, and the `CUDA_VISIBLE_DEVICES=...` line confirms
you actually hit the slice — if it logs `(unset)`, your variable didn't take.
The most common cause is a singular/plural typo: it must be
`CUDA_VISIBLE_DEVICES`, with the trailing **S**.

### File transfer

```bash
admin@wenlab-workstation:~$ scp file.txt alice@wenlab-workstation.local:~/          # single file
admin@wenlab-workstation:~$ rsync -avP folder/ alice@...:~/dest/                     # large/many files
admin@wenlab-workstation:~$ sftp alice@wenlab-workstation.local                      # interactive (put/get)
admin@wenlab-workstation:~$ ssh-copy-id alice@...                                    # passwordless thereafter
```

SFTP is built into the SSH server — no extra setup. GUI clients (FileZilla,
WinSCP, Cyberduck, or `sftp://` in GNOME Files) use the same backend.

### GUI for users

- **One-off GUI app:** `ssh -X user@host` then launch it — zero setup.
- **Full remote desktops (multiple simultaneous users):** install a light DE and
  a remote-desktop server.

  ```bash
  admin@wenlab-workstation:~$ sudo apt install xfce4 xfce4-goodies     # keep gdm3 as display manager when prompted
  ```

  - **x2go** (rides over SSH/port 22): `sudo apt install x2goserver
    x2goserver-xsession` (may need `ppa:x2go/stable`). Client via package
    manager — `winget install X2go.x2goclient` (Win),
    `brew install --cask x2goclient xquartz` (Mac), `sudo apt install x2goclient`
    (Linux).
  - **xrdp** (easier for Windows users — RDP is built into Windows):
    `sudo apt install xrdp && sudo systemctl enable --now xrdp`, then
    `echo "xfce4-session" > ~/.xsession` per user. Listens on port 3389.

  GUIs use software rendering (fine for IDEs/plots); GPU compute is unaffected.

---

## 8. Monitoring (real-time)

```bash
admin@wenlab-workstation:~$ htop                 # CPU / memory / processes
admin@wenlab-workstation:~$ nvitop               # GPU + MIG instances (pip install nvitop)  ← best for ML
admin@wenlab-workstation:~$ nvidia-smi -l 1      # GPU snapshot, refreshing
admin@wenlab-workstation:~$ sensors              # temperatures (apt install lm-sensors; run sensors-detect)
admin@wenlab-workstation:~$ iostat -x 1          # disk I/O (apt install sysstat)
admin@wenlab-workstation:~$ glances              # all-in-one dashboard
admin@wenlab-workstation:~$ watch -n 1 <cmd>     # turn any snapshot command into a live monitor
```

Recommended day-to-day: `htop` + `nvitop` in two tmux panes.

---

## Outstanding to-dos

- [x] Boot-time systemd service to recreate MIG instances after reboot — **done** (see [Persist MIG instances across reboots](#persist-mig-instances-across-reboots)).
- [ ] Decide MIG layout (4×24 vs 48+24+24) based on actual job profiles.
- [ ] Hand users their per-slice `CUDA_VISIBLE_DEVICES` (or set up `genv`).
