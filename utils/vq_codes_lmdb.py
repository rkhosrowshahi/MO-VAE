"""
LMDB-backed storage and Dataset for pre-extracted VQ-VAE discrete codes.
Avoids re-running VQ-VAE on images every epoch during prior training.
"""

import os
import pickle
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False


def extract_codes_to_lmdb(
    net: torch.nn.Module,
    train_loader,
    device: torch.device,
    lmdb_path: str,
    is_hierarchical: bool,
    map_size: int = 150 * 1024 * 1024 * 1024,  # 150GB default for ImageNet
) -> str:
    """
    One-time extraction: run all training samples through VQ-VAE and store codes in LMDB.
    
    Args:
        net: Frozen VQ-VAE model
        train_loader: DataLoader of (images, labels)
        device: Device for VQ-VAE forward
        lmdb_path: Directory path for LMDB database
        is_hierarchical: True for VQVAE2/GGVQVAE2 (z_top, z_bottom), False for single-level (z)
        map_size: LMDB map size in bytes (increase for large datasets like ImageNet)
    
    Returns:
        Path to the created LMDB database
    """
    if not HAS_LMDB:
        raise ImportError("lmdb is required for pre-extracted codes. Install with: pip install lmdb")
    
    os.makedirs(lmdb_path, exist_ok=True)
    net.eval()
    
    idx = 0
    meta = None
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        writemap=True,
        map_async=True,
    )
    
    tqdm = __import__("tqdm", fromlist=["tqdm"]).tqdm
    with torch.no_grad():
        for images, _ in tqdm(train_loader, desc="Extracting VQ codes to LMDB"):
            images = images.to(device)
            if is_hierarchical:
                code_dict = net.get_code_indices(images)
                z_top = code_dict["indices_top"].cpu().numpy()
                z_bottom = code_dict["indices_bottom"].cpu().numpy()
                if meta is None:
                    meta = {
                        "is_hierarchical": True,
                        "z_top_shape": tuple(z_top.shape[1:]),
                        "z_bottom_shape": tuple(z_bottom.shape[1:]),
                        "dtype": "int64",
                    }
                    with env.begin(write=True) as txn:
                        txn.put(b"__meta__", pickle.dumps(meta))
                for i in range(images.size(0)):
                    key = f"{idx}".encode()
                    val = pickle.dumps((z_top[i], z_bottom[i]))
                    with env.begin(write=True) as txn:
                        txn.put(key, val)
                    idx += 1
            else:
                z = net.get_code_indices(images)
                z_np = z.cpu().numpy()
                if meta is None:
                    meta = {
                        "is_hierarchical": False,
                        "z_shape": tuple(z_np.shape[1:]),
                        "dtype": "int64",
                    }
                    with env.begin(write=True) as txn:
                        txn.put(b"__meta__", pickle.dumps(meta))
                for i in range(images.size(0)):
                    key = f"{idx}".encode()
                    val = pickle.dumps((z_np[i],))
                    with env.begin(write=True) as txn:
                        txn.put(key, val)
                    idx += 1
    
    with env.begin(write=True) as txn:
        txn.put(b"__len__", str(idx).encode())
    
    env.sync()
    env.close()
    return lmdb_path


class VQCodeLMDBDataset(Dataset):
    """
    PyTorch Dataset that loads pre-extracted VQ codes from LMDB.
    Supports both single-level (z) and hierarchical (z_top, z_bottom) codes.
    """
    def __init__(self, lmdb_path: str, is_hierarchical: bool, transform_index: Optional[callable] = None):
        """
        Args:
            lmdb_path: Path to LMDB database (directory)
            is_hierarchical: True if stored as (z_top, z_bottom), False if (z,)
            transform_index: Optional callable to transform code indices (e.g., augment)
        """
        if not HAS_LMDB:
            raise ImportError("lmdb is required. Install with: pip install lmdb")
        
        self.lmdb_path = lmdb_path
        self.is_hierarchical = is_hierarchical
        self.transform_index = transform_index
        
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        
        with self.env.begin() as txn:
            meta_bytes = txn.get(b"__meta__")
            if meta_bytes is None:
                raise ValueError(f"LMDB at {lmdb_path} has no metadata (missing __meta__)")
            self.meta = pickle.loads(meta_bytes)
            
            len_bytes = txn.get(b"__len__")
            self._len = int(len_bytes.decode()) if len_bytes else 0
        
        # Validate
        if self.meta.get("is_hierarchical") != is_hierarchical:
            raise ValueError(
                f"LMDB is_hierarchical={self.meta.get('is_hierarchical')} "
                f"but requested is_hierarchical={is_hierarchical}"
            )
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, idx: int):
        with self.env.begin() as txn:
            val = txn.get(f"{idx}".encode())
            if val is None:
                raise IndexError(f"Index {idx} not found in LMDB")
            codes = pickle.loads(val)
        
        if self.is_hierarchical:
            z_top, z_bottom = codes
            z_top = torch.from_numpy(z_top).long()
            z_bottom = torch.from_numpy(z_bottom).long()
            if self.transform_index is not None:
                z_top = self.transform_index(z_top)
                z_bottom = self.transform_index(z_bottom)
            return z_top, z_bottom
        else:
            (z,) = codes
            z = torch.from_numpy(z).long()
            if self.transform_index is not None:
                z = self.transform_index(z)
            return z
    
    def close(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
    
    def __del__(self):
        self.close()


def get_or_extract_codes_lmdb(
    net: torch.nn.Module,
    train_loader,
    device: torch.device,
    save_root: str,
    is_hierarchical: bool,
    args,
    force_extract: bool = False,
    map_size: int = 150 * 1024 * 1024 * 1024,
) -> Tuple[Optional[Dataset], bool]:
    """
    Get VQCodeLMDBDataset, extracting codes to LMDB first if not present.
    
    Returns:
        (dataset, used_lmdb): dataset to use for prior training; used_lmdb=True if LMDB was used
    """
    # Config hash for path uniqueness (model + dataset identify the codes)
    input_size = getattr(net, "input_size", getattr(args, "input_size", 256))
    config_str = (
        f"{getattr(args, 'arch', 'vae')}_{getattr(args, 'dataset', '')}_"
        f"{getattr(net, 'num_embeddings', 512)}_{input_size}"
    )
    import hashlib
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    lmdb_path = os.path.join(save_root, "vq_codes_lmdb", config_hash)
    
    use_lmdb = getattr(args, "prior_use_lmdb_codes", True)
    
    if not use_lmdb:
        return None, False
    
    if not HAS_LMDB:
        tqdm = __import__("tqdm", fromlist=["tqdm"]).tqdm
        tqdm.write("Warning: lmdb not installed. Install with 'pip install lmdb' to use pre-extracted codes.")
        return None, False
    
    # Check if LMDB exists and is valid
    if not force_extract and os.path.isdir(lmdb_path):
        try:
            ds = VQCodeLMDBDataset(lmdb_path, is_hierarchical)
            if len(ds) > 0:
                tqdm = __import__("tqdm", fromlist=["tqdm"]).tqdm
                tqdm.write(f"Using pre-extracted codes from LMDB: {lmdb_path} ({len(ds)} samples)")
                return ds, True
        except Exception as e:
            tqdm = __import__("tqdm", fromlist=["tqdm"]).tqdm
            tqdm.write(f"LMDB load failed ({e}), will extract codes...")
    
    # Extract codes
    tqdm = __import__("tqdm", fromlist=["tqdm"]).tqdm
    tqdm.write(f"Extracting VQ codes to LMDB (one-time): {lmdb_path}")
    extract_codes_to_lmdb(
        net, train_loader, device, lmdb_path,
        is_hierarchical=is_hierarchical, map_size=map_size,
    )
    tqdm.write(f"Extraction complete. {lmdb_path}")
    
    ds = VQCodeLMDBDataset(lmdb_path, is_hierarchical)
    return ds, True
