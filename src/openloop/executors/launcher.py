"""POSIX limits for reviewed local commands. This is not a security sandbox.

Set limits in a fresh interpreter, then exec. Do not use preexec_fn in the
coordinator: it can deadlock when another coordinator thread holds a lock.
"""

import os
import resource
import sys


def main() -> None:
    # "-" leaves a limit unset.
    memory = int(sys.argv[1])
    cpu = None if sys.argv[2] == "-" else int(sys.argv[2])
    file_size = None if sys.argv[3] == "-" else int(sys.argv[3])
    # Darwin exposes RLIMIT_AS but rejects attempts to set it. The coordinator
    # monitors aggregate resident memory on that platform instead.
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    if cpu is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if file_size is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
    os.execvpe(sys.argv[4], sys.argv[4:], os.environ)


if __name__ == "__main__":
    main()
