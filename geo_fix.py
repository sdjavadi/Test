import glob as _glob
all_paths = ["file://" + os.path.abspath(p) for p in sorted(_glob.glob(EDGE_GLOB))]
if not all_paths:
    raise FileNotFoundError(f"{EDGE_GLOB} matched nothing. cwd={os.getcwd()}")
print(f"{len(all_paths)} snapshot files found (local)")
