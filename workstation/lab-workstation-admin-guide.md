# Research Lab Workstation — Administrator Setup Guide

For the **workstation manager**. Everything here requires `sudo`/root and is
one-time configuration to set up and configure the shared GPU workstation
(2× RTX PRO 6000 Blackwell Max-Q, 96 GB each; Ubuntu 24.04). Regular users
should follow the separate **User Guide** for day-to-day use.

## Table of Contents

- [1. User accounts and SSH](#1-user-accounts-and-ssh)
  - [Create user accounts](#create-user-accounts)
  - [Enable the SSH server](#enable-the-ssh-server)
  - [Shared directory](#shared-directory)
  - [Rename the machine](#rename-the-machine-optional)
- [2. Resource limits](#2-resource-limits)
  - [CPU and memory limits](#cpu-and-memory-limits-systemd-slices)
  - [Disk quotas](#disk-quotas)
- [3. NVIDIA driver](#3-nvidia-driver)
- [4. MIG (Multi-Instance GPU)](#4-mig-multi-instance-gpu)
  - [The hybrid layout](#the-hybrid-layout)
  - [Prerequisites](#prerequisites)
  - [Switch to compute mode](#switch-the-target-card-to-compute-mode)
  - [Enable MIG and create instances](#enable-mig-and-create-instances)
  - [Persist MIG instances across reboots](#persist-mig-instances-across-reboots)
  - [Troubleshooting](#troubleshooting-instances-missing-after-reboot)
  - [Assign slices to users](#assign-slices-to-users)
- [5. Shared software](#5-shared-software)
- [6. Shared conda (system-wide)](#6-shared-conda-system-wide)
- [7. GUI server setup](#7-gui-server-setup)
- [8. Monitoring](#8-monitoring)
- [Outstanding to-dos](#outstanding-to-dos)

---

## 1. User accounts and SSH

### Create user accounts

One account per person — never share a login.

```bash
admin@wenlab-workstation:~$ sudo adduser alice
admin@wenlab-workstation:~$ sudo usermod -aG sudo alice    # only for admins who need sudo
```

### Enable the SSH server

This is the backbone for 3–4 simultaneous users.

```bash
admin@wenlab-workstation:~$ sudo apt update
admin@wenlab-workstation:~$ sudo apt install openssh-server
admin@wenlab-workstation:~$ sudo systemctl enable --now ssh
```

Find the address users will connect to:

```bash
admin@wenlab-workstation:~$ hostname -I            # first address is usually the one to use
admin@wenlab-workstation:~$ ip route get 1.1.1.1   # the "src" value is the active IP
```

Set a **static IP or DHCP reservation** so the address doesn't change on reboot,
then give users that address (and their username). SFTP is included in the SSH
server automatically — no extra setup for file transfer. For the step-by-step
client instructions you'll hand to users, see the **User Guide**.

### Shared directory

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

---

## 2. Resource limits

So one user can't starve the box.

### CPU and memory limits (systemd slices)

```bash
# UID 1001 example: cap at ~200% CPU (2 cores) and 32 GB RAM
admin@wenlab-workstation:~$ sudo systemctl set-property user-1001.slice CPUQuota=200% MemoryMax=32G
```

To revert (empty = reset to unlimited):

```bash
admin@wenlab-workstation:~$ sudo systemctl set-property user-1001.slice CPUQuota="" MemoryMax=""
```

### Disk quotas

> **Hard-won lesson:** a typo in `/etc/fstab` (`defaut` instead of `defaults`)
> caused a boot failure. **Always double-check spelling when editing a config file.**

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

## 4. MIG (Multi-Instance GPU)

The GPU-sharing strategy.

### The hybrid layout

(recommended for a shared lab)

- **GPU 0:** left in normal mode → a whole 96 GB card; also drives the display;
  used for big single jobs needing the full card.
- **GPU 1:** partitioned with MIG into isolated slices (e.g. four × 24 GB) →
  one dedicated slice per user, hardware-isolated (no OOM-ing each other).

MIG memory comes in fixed **24 GB slices** (4 per card), so valid sizes are
multiples of 24: 24, 48, or 96 GB. No 32 GB option. For 3 instances you choose
between `3×24` (one slice idle) or `48+24+24`.

### Prerequisites

(all required)

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

### Troubleshooting: instances missing after reboot

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

### Assign slices to users

Each MIG instance is addressed by a `MIG-...` UUID. List them, then hand each
user their own so they never have to think about UUIDs:

```bash
admin@wenlab-workstation:~$ nvidia-smi -L        # lists the four MIG-... UUIDs under GPU 1
```

Set each user's assigned UUID in their shell profile (as that user, or edit
`/home/alice/.bashrc` as admin):

```bash
# in alice's ~/.bashrc:
export CUDA_VISIBLE_DEVICES=MIG-<alice-uuid>
```

After that they simply run jobs and land on their own isolated slice. (See the
**User Guide** for how users run on their slice.) For dynamic allocation instead
of fixed assignments, install `genv`.

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

## 6. Shared conda (system-wide)

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

**Policy:** keep shared read-only base envs in `/opt/conda/envs` for common
stacks; users create their own private envs in their home dirs (see the **User
Guide**). CUDA comes per-env via conda/pip, so no system CUDA toolkit needed.

---

## 7. GUI server setup

To let users run graphical apps remotely, install a lightweight desktop and a
remote-desktop server (the *client* side is in the User Guide).

```bash
admin@wenlab-workstation:~$ sudo apt install xfce4 xfce4-goodies     # keep gdm3 as display manager when prompted
```

Then pick a server:

- **x2go** (rides over SSH/port 22):

  ```bash
  admin@wenlab-workstation:~$ sudo apt install x2goserver x2goserver-xsession   # may need ppa:x2go/stable
  ```

- **xrdp** (easier for Windows users — RDP is built into Windows; listens on port 3389):

  ```bash
  admin@wenlab-workstation:~$ sudo apt install xrdp
  admin@wenlab-workstation:~$ sudo systemctl enable --now xrdp
  ```

  Each user then runs once: `echo "xfce4-session" > ~/.xsession`.

GUIs use software rendering (fine for IDEs/plots); GPU compute is unaffected. For
single one-off apps, users can also just use `ssh -X` with no server at all.

---

## 8. Monitoring

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
