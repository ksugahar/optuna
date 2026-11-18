#!/bin/bash
# Stage a Flow2 job root on the login node: exact source, hash-locked wheelhouses,
# owner.json and inputs.sha256. Compute jobs then run without network access.
# Usage: stage.sh SOURCE SHA JOB_ROOT OWNER REQUEST DEADLINE_HOURS
#   SOURCE  git URL or bundle path that contains SHA
set -Eeuo pipefail
[ $# -eq 6 ] || { sed -n '2,5p' "$0"; exit 2; }
source=$1 sha=$2 root=$3 owner=$4 request=$5 hours=$6
[[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "SHA must be a full 40-hex commit" >&2; exit 2; }
mkdir "$root"   # fails if the root exists: never reuse or overwrite a job root
umask 077
git clone -q --no-checkout "$source" "$root/source"
git -C "$root/source" checkout -q --detach "$sha"
test "$(git -C "$root/source" rev-parse HEAD)" = "$sha"
here=$root/source/radia-optuna/tools/flow2
platform="--only-binary=:all: --implementation cp --python-version 3.12 --abi cp312
  --platform manylinux_2_28_aarch64 --platform manylinux_2_27_aarch64
  --platform manylinux_2_24_aarch64 --platform manylinux_2_17_aarch64 --platform manylinux2014_aarch64"
for set in build validate; do
  # shellcheck disable=SC2086
  python3.12 -m pip download -q $platform --no-deps --require-hashes \
    -r "$here/requirements-$set.txt" -d "$root/wheelhouse-$set"
done
python3.12 - "$root" "$owner" "$request" "$sha" "$hours" <<'PY'
import datetime, json, sys
root, owner, request, sha, hours = sys.argv[1:]
now = datetime.datetime.now(datetime.timezone.utc)
json.dump({"owner": owner, "request": request, "source_sha": sha,
           "created_utc": now.isoformat(),
           "deadline_utc": (now + datetime.timedelta(hours=float(hours))).isoformat(),
           "scope": "Flow2 Linux Python GPU wheel build and installed-wheel GPU checks; not a performance study"},
          open(f"{root}/owner.json", "w"), indent=2)
PY
(cd "$root" && find wheelhouse-build wheelhouse-validate -type f | sort | xargs sha256sum > inputs.sha256)
echo "staged $root at $sha"
