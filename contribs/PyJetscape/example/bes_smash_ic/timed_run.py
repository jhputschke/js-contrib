#!/usr/bin/env python3
"""timed_run.py LOG TIMELOG -- CMD [ARGS...]

Runs CMD, writes its stdout and stderr to LOG with every line prefixed by the
seconds since the start, and writes the wall time, the peak resident memory
and the exit status to TIMELOG (in the words /usr/bin/time -v uses, so the
same grep works). Python only: the production image has neither /usr/bin/time
nor Perl's Time::HiRes.
"""
import resource
import subprocess
import sys
import time


def main():
    if len(sys.argv) < 5 or sys.argv[3] != "--":
        sys.exit(__doc__)
    log, timelog, cmd = sys.argv[1], sys.argv[2], sys.argv[4:]
    t0 = time.time()
    with open(log, "w") as out:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1,
                                text=True, errors="replace")
        for line in proc.stdout:
            out.write("%9.3f %s" % (time.time() - t0, line))
            out.flush()
        status = proc.wait()
    wall = time.time() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    m, s = divmod(wall, 60)
    with open(timelog, "w") as f:
        f.write("\tCommand being timed: \"%s\"\n" % " ".join(cmd))
        f.write("\tElapsed (wall clock) time (h:mm:ss or m:ss): %d:%05.2f\n"
                % (m, s))
        f.write("\tMaximum resident set size (kbytes): %d\n" % rss_kb)
        f.write("\tExit status: %d\n" % status)
    sys.exit(status)


if __name__ == "__main__":
    main()
