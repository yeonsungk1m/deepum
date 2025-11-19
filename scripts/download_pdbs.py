import requests
from pathlib import Path

ids_file = Path("pisces_parsed_ids.txt")
out_dir = Path("data/train/native_pdb_all")
out_dir.mkdir(parents=True, exist_ok=True)

pdb_ids = sorted({line.split()[0] for line in ids_file.read_text().splitlines() if line.strip()})

base_url = "https://files.rcsb.org/download"

for pdb_id in pdb_ids:
    out_path = out_dir / f"{pdb_id}.pdb"
    if out_path.exists():
        continue
    url = f"{base_url}/{pdb_id.upper()}.pdb"
    r = requests.get(url)
    if r.status_code == 200:
        out_path.write_bytes(r.content)
    else:
        print(f"Failed to download {pdb_id}: HTTP {r.status_code}")
