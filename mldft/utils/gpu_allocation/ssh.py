"""Shared SSH helpers for cluster commands."""

from __future__ import annotations

SSH_OPTIONS: tuple[str, ...] = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=5",
)


def ssh_setup_hint(host: str) -> str:
    """Return user-facing guidance for SSH hostname or key setup failures."""
    return (
        "Remote GPU fallback expects fallback hosts to be reachable as direct "
        "SSH targets. "
        f"Make sure `ssh {host}` works on this machine. If it does not, create "
        "an SSH key, for example called `cluster`, and configure your SSH client.\n\n"
        "Create the key with:\n"
        "    ssh-keygen -t ed25519 -f ~/.ssh/cluster\n\n"
        "Then add a config like this to `~/.ssh/config`:\n"
        "    Host compgpu*\n"
        "        IdentityFile ~/.ssh/cluster\n"
        "        User <your-cluster-user>\n"
        "        ForwardAgent yes\n\n"
        "Add the host key to `~/.ssh/known_hosts` as usual, for example:\n"
        f"    ssh-keyscan -t ed25519 {host} >> ~/.ssh/known_hosts"
    )
