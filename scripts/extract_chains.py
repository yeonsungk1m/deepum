from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select

ids_file = Path("pisces_parsed_ids.txt")
pdb_dir = Path("data/train/native_pdb_all")
out_dir = Path("data/train/native_chains")
out_dir.mkdir(parents=True, exist_ok=True)

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id

parser = PDBParser(QUIET=True)

with ids_file.open() as f:
    for line in f:
        if not line.strip():
            continue
        pdb_id, chain_id = line.split()
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            print(f"Skip {pdb_id}{chain_id}: PDB not found")
            continue

        structure = parser.get_structure(f"{pdb_id}", pdb_path)
        # 첫 model 사용
        model = next(structure.get_models())
        chains = [c for c in model.get_chains() if c.id == chain_id]
        if not chains:
            print(f"Chain {chain_id} not found in {pdb_id}")
            continue

        io = PDBIO()
        io.set_structure(model)
        out_path = out_dir / f"{pdb_id}_{chain_id}.pdb"
        io.save(str(out_path), ChainSelect(chain_id))

        # 나중에 native 이름 통일을 위해 log
        # print("Wrote", out_path)
