from pathlib import Path

src = Path("cullpdb_pc25.0_res0.0-2.0_len40-300_R0.25_Xray_d2025_11_11_chains6373")
out = Path("pisces_parsed_ids.txt")

chains = []
with src.open() as f:
    header = next(f)  # skip header
    for line in f:
        if not line.strip():
            continue
        cols = line.split()
        pdb_chain = cols[0]  # e.g., 5D8VA
        pdb_id = pdb_chain[:4].lower()
        chain_id = pdb_chain[4:]  # e.g., 'A'
        chains.append((pdb_id, chain_id))

with out.open("w") as f:
    for pdb_id, chain_id in chains:
        f.write(f"{pdb_id} {chain_id}\n")

print(f"Parsed {len(chains)} chains -> {out}")
