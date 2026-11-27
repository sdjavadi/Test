#!/usr/bin/env python
"""
WebHDFS probe, part 2 of 2 — runs on the GPU node (sbatch pkg_webhdfs_probe.sh); the log is the output.
Question: can the GPU node read files in your HDFS space, so exports can stay on HDFS instead of the shared folder?
Reads ../net_work/_webhdfs_probe.json (written by the notebook cell), then checks in order:
  1. Kerberos ticket on this node (klist)
  2. which namenode web addresses answer at all (TCP)
  3. curl --negotiate: list the test folder and read the test file (WebHDFS REST)
  4. Python: requests with Kerberos (requests_gssapi / requests_kerberos, whichever is installed), then read the
     small parquet folder into pandas
  5. pyarrow's own HDFS client (needs libhdfs and a Hadoop client; expected to fail here, reported for completeness)
"""
import os, sys, json, socket, subprocess, shutil, io, time
SHARED = os.environ.get("PKG_NET_DIR") or os.path.abspath("../net_work")

def head(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90, flush=True)
def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"

head(f"host {socket.gethostname()} · job {os.environ.get('SLURM_JOB_ID', '-')} · python {sys.version.split()[0]}")
info = json.load(open(os.path.join(SHARED, "_webhdfs_probe.json")))
conf = info["conf"]
for k in sorted(conf):
    print(f"  {k} = {conf[k]}")

head("1. Kerberos ticket")
rc, o = sh("klist"); print(o or f"klist rc {rc}")
has_ticket = rc == 0
if not has_ticket:
    print("No ticket on this node. Steps 3–4 will fail with 401 until a ticket exists here (kinit in the sbatch script, "
          "or a keytab).")

head("2. Namenode web addresses that answer")
cands = []
for k, v in conf.items():
    if ("http-address" in k or "https-address" in k) and ":" in str(v) and "0.0.0.0" not in str(v):
        scheme = "https" if "https-address" in k else "http"
        cands.append((scheme, v.strip()))
for k, v in conf.items():
    if "httpfs" in k and ":" in str(v):
        cands.append(("http", v.strip()))
cands = sorted(set(cands))
alive = []
for scheme, hp in cands:
    h, p = hp.rsplit(":", 1)
    try:
        socket.create_connection((h, int(p)), timeout=5).close(); ok = True
    except Exception as e:
        ok = False; err = type(e).__name__
    print(f"  {scheme}://{hp:45s} {'open' if ok else 'closed (' + err + ')'}")
    if ok:
        alive.append(f"{scheme}://{hp}")
if not cands:
    print("  no namenode web address in the Hadoop settings")

head("3. curl --negotiate (WebHDFS REST)")
works = None
if shutil.which("curl") is None:
    print("  curl not found")
for base in alive:
    for op, path in (("LISTSTATUS", os.path.dirname(info["hdfs_path"])), ("OPEN", info["hdfs_path"])):
        url = f"{base}/webhdfs/v1{path}?op={op}"
        rc, o = sh(f"curl -s -k -L --negotiate -u : -w ' [HTTP %{{http_code}}]' '{url}'", timeout=40)
        print(f"  {op:10s} {base}: {o[:300]}")
        if op == "OPEN" and "hello from the notebook host" in o:
            works = base
print(f"\n  → WebHDFS by curl: {'WORKS via ' + works if works else 'not working'}")

head("4. Python requests + Kerberos, then the parquet folder into pandas")
auth = None
try:
    from requests_gssapi import HTTPSPNEGOAuth; auth = HTTPSPNEGOAuth(); print("  using requests_gssapi")
except Exception:
    try:
        from requests_kerberos import HTTPKerberosAuth, OPTIONAL; auth = HTTPKerberosAuth(mutual_authentication=OPTIONAL); print("  using requests_kerberos")
    except Exception:
        print("  neither requests_gssapi nor requests_kerberos is installed (pip install requests-gssapi)")
if auth is not None and alive:
    import requests, pandas as pd
    for base in ([works] if works else alive):
        try:
            t = time.time()
            r = requests.get(f"{base}/webhdfs/v1{info['parquet_dir']}?op=LISTSTATUS", auth=auth, verify=False, timeout=30)
            print(f"  LISTSTATUS {base}: HTTP {r.status_code}")
            if r.ok:
                files = [f["pathSuffix"] for f in r.json()["FileStatuses"]["FileStatus"] if f["pathSuffix"].endswith(".parquet")]
                parts = []
                for f in files:
                    rr = requests.get(f"{base}/webhdfs/v1{info['parquet_dir']}/{f}?op=OPEN", auth=auth, verify=False, timeout=60)
                    rr.raise_for_status(); parts.append(pd.read_parquet(io.BytesIO(rr.content)))
                df = pd.concat(parts)
                print(f"  read {len(files)} parquet file(s), {len(df):,} rows (expected 1,000) in {time.time() - t:.1f} s")
                break
        except Exception as e:
            print(f"  {base}: {type(e).__name__}: {str(e)[:200]}")

head("5. pyarrow HDFS client")
try:
    import pyarrow.fs as pafs
    h = pafs.HadoopFileSystem(info["authority"].split(":")[0] if info.get("authority") else "default")
    print("  connected:", h.get_file_info(info["hdfs_path"]))
except Exception as e:
    print(f"  not available: {type(e).__name__}: {str(e)[:200]}")

head("VERDICT")
print("  GPU node can read your HDFS space over WebHDFS" if works else
      "  GPU node cannot read HDFS yet — see steps 1–3 for which part blocks it")
