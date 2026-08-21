"""Node identity, registration PoW, and signed credit receipts (Sec. 5.4)."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature


def canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class NodeIdentity:
    """Ed25519 node identity derived at first run (Sec. 5.4)."""

    public_key_hex: str
    _private_key: Ed25519PrivateKey | None = field(default=None, repr=False)

    @classmethod
    def generate(cls) -> NodeIdentity:
        key = Ed25519PrivateKey.generate()
        pub = key.public_key().public_bytes_raw().hex()
        return cls(public_key_hex=pub, _private_key=key)

    @classmethod
    def from_public_hex(cls, public_key_hex: str) -> NodeIdentity:
        return cls(public_key_hex=public_key_hex, _private_key=None)

    @property
    def public_key_bytes(self) -> bytes:
        return bytes.fromhex(self.public_key_hex)

    @property
    def node_id(self) -> str:
        """Stable node id from public key (DHT advertisement key)."""
        return hashlib.sha256(self.public_key_bytes).hexdigest()[:32]

    def sign(self, payload: dict[str, Any]) -> str:
        if self._private_key is None:
            raise ValueError("private key required to sign")
        sig = self._private_key.sign(canonical_json(payload))
        return sig.hex()

    def verify(self, payload: dict[str, Any], signature_hex: str) -> bool:
        try:
            pub = Ed25519PublicKey.from_public_bytes(self.public_key_bytes)
            pub.verify(bytes.fromhex(signature_hex), canonical_json(payload))
            return True
        except (InvalidSignature, ValueError):
            return False


def registration_pow_digest(node_id: str, nonce: int) -> bytes:
    return hashlib.sha256(f"swarmharness:register:{node_id}:{nonce}".encode()).digest()


def verify_registration_pow(node_id: str, nonce: int, difficulty_bits: int) -> bool:
    """Lightweight SHA-256 PoW for Sybil resistance (Sec. 5.4)."""
    if difficulty_bits <= 0:
        return True
    if difficulty_bits > 256:
        raise ValueError("difficulty_bits must be <= 256")
    digest = registration_pow_digest(node_id, nonce)
    value = int.from_bytes(digest, "big")
    return value < (1 << (256 - difficulty_bits))


def mine_registration_pow(node_id: str, difficulty_bits: int, *, max_tries: int = 500_000) -> int:
    """Find a nonce satisfying verify_registration_pow (for tests / local join)."""
    for nonce in range(max_tries):
        if verify_registration_pow(node_id, nonce, difficulty_bits):
            return nonce
    raise RuntimeError(f"PoW not found within {max_tries} tries at {difficulty_bits} bits")


@dataclass(frozen=True)
class CreditReceipt:
    """Submitter-signed credit attribution (Sec. 3.4, 5.4)."""

    task_id: str
    submitter_id: str
    contributors: tuple[str, ...]
    deltas: dict[str, float]
    credit_pool: float
    quality: float
    issued_at: float

    def payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "submitter_id": self.submitter_id,
            "contributors": list(self.contributors),
            "deltas": {k: round(v, 9) for k, v in sorted(self.deltas.items())},
            "credit_pool": round(self.credit_pool, 9),
            "quality": round(self.quality, 9),
            "issued_at": self.issued_at,
        }

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        submitter_id: str,
        contributors: list[str],
        deltas: dict[str, float],
        credit_pool: float,
        quality: float,
        issued_at: float | None = None,
    ) -> CreditReceipt:
        return cls(
            task_id=task_id,
            submitter_id=submitter_id,
            contributors=tuple(contributors),
            deltas=dict(deltas),
            credit_pool=credit_pool,
            quality=quality,
            issued_at=issued_at if issued_at is not None else time.time(),
        )


def sign_credit_receipt(identity: NodeIdentity, receipt: CreditReceipt) -> dict[str, Any]:
    payload = receipt.payload()
    return {
        "receipt": payload,
        "submitter_pubkey": identity.public_key_hex,
        "signature": identity.sign(payload),
    }


def verify_credit_receipt(signed: dict[str, Any]) -> bool:
    pub_hex = str(signed.get("submitter_pubkey", ""))
    sig = str(signed.get("signature", ""))
    payload = signed.get("receipt")
    if not pub_hex or not sig or not isinstance(payload, dict):
        return False
    ident = NodeIdentity.from_public_hex(pub_hex)
    return ident.verify(payload, sig)
