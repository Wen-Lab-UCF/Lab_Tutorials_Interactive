# Research Lab Workstation — User Guide

For **regular users** of the shared GPU workstation. This covers connecting
remotely, running GPU jobs on your assigned slice, keeping jobs running after you
log out, transferring files, and using the GUI. You don't need `sudo` — your
account and GPU slice are set up by the workstation manager.

**Before you start, ask the workstation manager for:**

- your **username** (e.g. `alice`),
- the workstation's **address** (an IP like `192.168.1.42`, or
  `wenlab-workstation.local`),
- your assigned **GPU (MIG) UUID** (looks like `MIG-9f6a1ac8-...`).

Throughout, `alice@laptop` means a command on *your own machine*, and
`alice@wenlab-workstation` means a command *on the workstation* after you've
connected.

## Table of Contents

- [1. Connect via SSH](#1-connect-via-ssh)
- [2. Transfer files](#2-transfer-files)
- [3. Use the shared conda](#3-use-the-shared-conda)
- [4. Run jobs on your GPU slice](#4-run-jobs-on-your-gpu-slice)
- [5. Keep jobs running after you log out](#5-keep-jobs-running-after-you-log-out)
  - [Worked example: a logged training run](#worked-example-a-logged-training-run)
- [6. Use the GUI remotely](#6-use-the-gui-remotely)
- [7. Monitor your usage](#7-monitor-your-usage)

---

## 1. Connect via SSH

Connect from your own machine using your account name and the workstation's IP
(or `wenlab-workstation.local` if you're on the same network and mDNS works).
Replace `alice` with your username and `<ip>` with the address you were given.
This works directly when you're on the **same network** as the workstation (e.g.
the lab or campus Wi-Fi); from off-site you'll need a VPN into that network.

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

## 2. Transfer files

These commands run on **your own machine**, pointing at the workstation. Because
you authenticate as yourself, transferred files are owned by you automatically.

```bash
alice@laptop:~$ scp file.txt alice@wenlab-workstation.local:~/      # single file → your home dir
alice@laptop:~$ scp -r myfolder alice@wenlab-workstation.local:~/   # a whole folder
alice@laptop:~$ rsync -avP folder/ alice@<ip>:~/dest/               # large/many files (resumable)
alice@laptop:~$ sftp alice@wenlab-workstation.local                 # interactive: put / get
```

SFTP is built into the workstation's SSH server — no setup needed. If you prefer
a graphical tool, **FileZilla**, **WinSCP**, or **Cyberduck** all speak SFTP
(host = the workstation address, port 22, your login); on Linux you can also type
`sftp://alice@<ip>` into the GNOME Files location bar.

---

## 3. Use the shared conda

Conda is already installed system-wide, so the `conda` command works out of the
box once you're logged in:

```bash
alice@wenlab-workstation:~$ conda --version        # confirms it's available
```

Don't modify the shared base environment. Instead, create your **own** private
environment (it lives in your home directory and is yours alone):

```bash
alice@wenlab-workstation:~$ conda create -n myenv python=3.12
alice@wenlab-workstation:~$ conda activate myenv
alice@wenlab-workstation:~$ pip install numpy pandas matplotlib      # or: conda install ...
```

For GPU work, install a **Blackwell-compatible PyTorch** (these cards need a
CUDA 12.8+ build; an older one will fail with "no kernel image"):

```bash
alice@wenlab-workstation:~$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Quick check that PyTorch sees a GPU:

```bash
alice@wenlab-workstation:~$ python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. Run jobs on your GPU slice

You're assigned one **MIG slice** — an isolated 24 GB GPU that's yours alone, so
your jobs can't be slowed down or crashed by other users. You select it with the
`CUDA_VISIBLE_DEVICES` environment variable set to your MIG **UUID**; inside
Python the slice is always `cuda:0`.

First, find your UUID (or use the one the manager gave you):

```bash
alice@wenlab-workstation:~$ nvidia-smi -L          # lists the MIG-... UUIDs
```

Run your job pinned to your slice:

```bash
alice@wenlab-workstation:~$ CUDA_VISIBLE_DEVICES=MIG-<your-uuid> python train.py
```

In your code, refer to it as `cuda:0`:

```python
import torch
device = torch.device("cuda:0")   # the slice you exposed
```

> **Common mistake:** the variable is `CUDA_VISIBLE_DEVICES` — **plural**, with a
> trailing **S**. `CUDA_VISIBLE_DEVICE` (no S) is silently ignored, and your job
> lands on the wrong GPU. If your script prints `CUDA_VISIBLE_DEVICES=(unset)` or
> `nvidia-smi` shows your process on GPU 0 instead of your slice, that's the
> cause.

If the manager set your UUID in your `~/.bashrc`, you can skip the variable
entirely and just run `python train.py` — it'll use your slice automatically.

---

## 5. Keep jobs running after you log out

Closing your SSH session normally kills whatever you were running. To keep a job
going after you log out (or your Wi-Fi drops, or you close your laptop), run it
inside a **terminal multiplexer** — `tmux` is the standard choice.

**tmux — the recommended approach.** `tmux` runs a shell on the workstation that
lives independently of your SSH connection. Start a named session:

```bash
alice@wenlab-workstation:~$ tmux new -s train
```

Inside it, launch your job normally (e.g. `python train.py`). Then **detach** by
pressing `Ctrl+b`, releasing, then `d`. You're back at your normal shell and the
job keeps running on the workstation — you can now safely close SSH.

When you return, SSH back in and reattach:

```bash
alice@wenlab-workstation:~$ tmux ls                  # list your running sessions
alice@wenlab-workstation:~$ tmux attach -t train     # reattach to "train"
```

Other handy commands:

```bash
alice@wenlab-workstation:~$ tmux new -s data             # a second named session
alice@wenlab-workstation:~$ tmux kill-session -t train   # stop a session when done
```

If `tmux ls` prints `no server running on ...` (or `error connecting to ...`), it
just means you have no sessions right now — nothing is wrong. Note that a full
reboot of the workstation does end tmux sessions, so you'd start fresh after one.

**Can I just close the SSH window?** Yes — tmux survives a disconnect, so closing
your terminal or losing Wi-Fi won't kill the job; tmux detaches automatically.
The clean habit is to detach first (`Ctrl+b` `d`), then close SSH. The one thing
**not** to do is type `exit` *inside* the tmux session — that closes the session
and kills the job.

**nohup — for fire-and-forget jobs.** If you only want to launch something and
not watch it interactively:

```bash
alice@wenlab-workstation:~$ nohup python train.py > train.log 2>&1 &
```

The `&` backgrounds it, `nohup` makes it ignore the disconnect signal, and you
check progress in `train.log`. Simpler than tmux, but you can't reattach to a
live view.

### Worked example: a logged training run

A complete example: a small training script that **logs to a file**, run on your
MIG slice **inside tmux**, so you can disconnect and later confirm it kept
running. Save this as `train_demo.py`:

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

Run it in tmux, pinned to your slice:

```bash
alice@wenlab-workstation:~$ nvidia-smi -L                # copy your MIG-... UUID
alice@wenlab-workstation:~$ tmux new -s train
# --- you are now inside the tmux session ---
alice@wenlab-workstation:~$ CUDA_VISIBLE_DEVICES=MIG-9f6a1ac8-... python train_demo.py
```

Detach with `Ctrl+b` then `d`, then close SSH. Later, reconnect and watch the
log — the timestamps prove it kept training while you were gone:

```bash
alice@wenlab-workstation:~$ tail -f train.log
```

```
14:02:11  RUN START  pid=8661  CUDA_VISIBLE_DEVICES=MIG-9f6a1ac8-...
14:02:13  device: NVIDIA RTX PRO 6000 Blackwell Max-Q ... MIG 1g.24gb  (23.6 GiB)
14:02:15  epoch 0  step 0  loss 2.3041
14:02:41  epoch 1  step 0  loss 0.3922
...
14:11:50  RUN COMPLETE
```

Or jump back into the live session with `tmux attach -t train`. Two checks are
built in: timestamps spanning the time you were disconnected confirm tmux kept
the job alive, and the `CUDA_VISIBLE_DEVICES=...` line confirms you hit your slice
— if it logs `(unset)`, your variable didn't take (most often the missing `S`).

---

## 6. Use the GUI remotely

If you only need a **single graphical app** occasionally, use X11 forwarding —
nothing to install on the workstation:

```bash
alice@laptop:~$ ssh -X alice@<ip>      # then launch the app, e.g. `spyder`
```

For a **full remote desktop**, the workstation runs a remote-desktop server (set
up by the manager). Connect with the matching client on your machine:

- **x2go** — install the client, then in it set Host = the workstation address,
  Login = your username, Session type = **XFCE**, and connect:

  ```bash
  alice@laptop:~$ winget install X2go.x2goclient          # Windows
  alice@laptop:~$ brew install --cask x2goclient xquartz  # macOS (XQuartz needed)
  alice@laptop:~$ sudo apt install x2goclient             # Linux
  ```

- **xrdp** — on **Windows** just open the built-in **Remote Desktop Connection**,
  enter the workstation address, and log in (Mac/Linux: "Microsoft Remote
  Desktop" or "Remmina").

You get your own independent desktop, and several users can be connected at once.
GUIs use software rendering (fine for IDEs and plots); your GPU/training jobs are
unaffected.

---

## 7. Monitor your usage

```bash
alice@wenlab-workstation:~$ nvitop          # GPU + your MIG slice (pip install nvitop)
alice@wenlab-workstation:~$ nvidia-smi       # quick GPU snapshot
alice@wenlab-workstation:~$ htop             # CPU / memory / your processes
```

A handy habit: keep `nvitop` open in one tmux pane while your job trains in
another, so you can watch your slice's utilization and memory live.
