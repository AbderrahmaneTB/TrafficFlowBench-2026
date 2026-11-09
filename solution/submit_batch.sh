#!/bin/sh
# Wait for the Kaggle daily reset (00:00 UTC) and upload a batch of built submissions in order.
# usage: sh submit_batch.sh name1 "msg1" name2 "msg2" ...
export PATH="/c/Users/HP/AppData/Local/Programs/Python/Python312:/c/Users/HP/AppData/Local/Programs/Python/Python312/Scripts:$PATH"
cd "$(dirname "$0")"
# AT = UTC time to start (default: the next 00:02 UTC); a PC that sleeps through it submits as soon as it wakes
AT=${AT:-$(date -u -d "tomorrow 00:02" "+%Y-%m-%d %H:%M")}
T0=$(date -u -d "$AT" +%s)
echo "$(date) waiting for $AT UTC"
until [ "$(date -u +%s)" -ge "$T0" ]; do sleep 30; done
while [ $# -ge 2 ]; do
  echo "$(date) submitting $1"
  python -u submit.py "$1" "$2" 2>&1 | tr '\r' '\n' | grep -v "MB/s\|kB/s" | tail -2
  shift 2
  sleep 20
done
echo "$(date) batch done"
