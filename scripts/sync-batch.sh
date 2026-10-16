#!/bin/sh
# Fetch one immutable return namespace through the operator's host SSH connection.
set -eu
if [ "$#" -ne 3 ]; then
    echo "usage: $0 SSH_HOST REMOTE_BATCH_DIRECTORY LOCAL_RETURN_DIRECTORY" >&2
    exit 2
fi
host=$1
remote=$2
local_return=$(realpath "$3")
case "$host" in
    ''|-*|*[!A-Za-z0-9_.@-]*) echo 'Use an SSH config host alias.' >&2; exit 2 ;;
esac
case "$remote" in
    /*) ;;
    *) echo 'Use an absolute remote return path.' >&2; exit 2 ;;
esac
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT HUP INT TERM
scp -r -- "$host:$remote" "$staging/batch"
cd "$repo"
uv run --locked --no-sync python -m openloop.batch.receive "$staging/batch" "$local_return"
