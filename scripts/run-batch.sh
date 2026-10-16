#!/bin/sh
# Run a reviewed finite batch. Return files go to a durable host-mounted directory.
set -eu
if [ "$#" -ne 4 ]; then
    echo "usage: $0 IMAGE_SHA256 QUEUE_JSON STATE_DIRECTORY RETURN_DIRECTORY" >&2
    exit 2
fi
image=$1
case "$image" in
    sha256:*) ;;
    *) echo 'Use an immutable local image ID (sha256:...), not a tag.' >&2; exit 2 ;;
esac
worker_user=${OPENLOOP_WORKER_USER:-$(id -u):$(id -g)}
case "$worker_user" in
    0|0:*|root|root:*) echo 'Set OPENLOOP_WORKER_USER to a non-root uid:gid.' >&2; exit 2 ;;
esac
queue=$(realpath "$2")
state=$(realpath "$3")
returned=$(realpath "$4")
[ -f "$queue" ] && [ -d "$state" ] && [ -d "$returned" ]
# No environment forwarding, credentials, Docker socket, pull, or automatic pod launch.
exec docker run --rm --pull never --init \
    --user "$worker_user" --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges --pids-limit 256 \
    --memory "${OPENLOOP_CONTAINER_MEMORY:-8g}" --cpus "${OPENLOOP_CONTAINER_CPUS:-4}" \
    --tmpfs /tmp:rw,nosuid,nodev,size=256m \
    --mount "type=bind,src=$queue,dst=/queue.json,readonly" \
    --mount "type=bind,src=$state,dst=/work" \
    --mount "type=bind,src=$returned,dst=/return" \
    "$image" --image-id "$image" \
    --queue /queue.json --state /work --sync-to /return
