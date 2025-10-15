
import torch.nn as nn
import torch
from src.logging.logging_helpers import logi, log_once, get_logger
import deepspeed
from src.zero3.config import DS_CONFIG
from typing import Any

class MyDeepspeed():
    def __init__(self, context_encoder, predictor, gpu_rank, world_size):
        self.gpu_rank = gpu_rank
        self.world_size = world_size
        
        # Store predictor separately
        self.predictor = predictor
        
        # Only encoder parameters for DeepSpeed
        self.model_parameters = [p for p in context_encoder.parameters() if p.requires_grad]

        # Calculate pre-shard stats for encoder only
        self.pre_total_all = self._sum_params_module(context_encoder, trainable_only=False)
        self.pre_total_trn = self._sum_params_module(context_encoder, trainable_only=True)
        self.pre_global_numels = {name: p.numel() for name, p in context_encoder.named_parameters()}

        # Initialize DeepSpeed with ONLY the encoder
        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            model=context_encoder,
            model_parameters=self.model_parameters,  # Only encoder params
            config=DS_CONFIG,
        )

        # Create separate optimizer for predictor (not managed by DeepSpeed)
        predictor_params = [p for p in predictor.parameters() if p.requires_grad]
        self.predictor_optimizer = torch.optim.AdamW( # TODO: Make this also come from ds_config maybe?
            predictor_params,
            lr=0.0005,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        zero_on = self.engine.zero_optimization()
        zero_stage = self.engine.zero_optimization_stage()

        logi(
            f"DeepSpeed ZeRO enabled: {zero_on}, stage: {zero_stage}, dp_world_size: {getattr(self.engine, 'dp_world_size', self.world_size)}", 
            self.gpu_rank
        )
        assert zero_on and zero_stage == 3, f"Expected ZeRO stage 3, got: {zero_stage}"

    def get_engine_and_optim(self):
        # Return both optimizers and predictor
        return self.engine, self.optimizer, self.predictor, self.predictor_optimizer

    def log_gpu_memory_usage(self):

        logi(
            f"[pre-shard] total params (all): {self.pre_total_all:,}", self.gpu_rank)
        logi(
            f"[pre-shard] total params (trainable): {self.pre_total_trn:,}", self.gpu_rank)

        self.log_param_distribution_and_size(
            self.engine, self.engine.device, self.gpu_rank, self.world_size)
        self.log_gpu_memory_snapshot("post-init-models", self.gpu_rank)
        self.log_first_local_params(self.engine, self.gpu_rank, max_items=10)

    def _sum_params_module(self, module: nn.Module, trainable_only: bool = False) -> int:
        """Count total parameters in a module."""
        return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))

    def _local_shard_numel(self, engine: Any, module: nn.Module) -> int:
        """
        Count *this rank's* parameter shard numel.

        Uses DeepSpeed ds_tensor if present.

        Args:
            engine: DeepSpeed engine
            module: PyTorch module

        Returns:
            Number of parameters in this rank's shard
        """
        local = 0
        for p in module.parameters():
            ds_tensor = getattr(p, "ds_tensor", None)
            if ds_tensor is not None:
                local += int(ds_tensor.numel())
            else:
                local += int(p.numel())
        return local

    def _local_shard_bytes(self, engine: Any, module: nn.Module) -> int:
        """
        Exact bytes for this rank's param shards.

        Accounts for mixed dtypes.

        Args:
            engine: DeepSpeed engine
            module: PyTorch module

        Returns:
            Total bytes of parameters in this rank's shard
        """
        total_bytes = 0
        for p in module.parameters():
            ds_tensor = getattr(p, "ds_tensor", None)
            n = int(ds_tensor.numel()) if ds_tensor is not None else int(
                p.numel())
            total_bytes += n * p.element_size()
        return total_bytes

    def _all_gather_i64(self, value: int, device: torch.device) -> list[int]:
        """
        All-gather a single int64 from each rank.

        Args:
            value: Integer value to gather
            device: Device to create tensors on

        Returns:
            List of integers from all ranks (or single value if not distributed)
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([int(value)], dtype=torch.long, device=device)
            outs = [torch.zeros_like(t) for _ in range(
                torch.distributed.get_world_size())]
            torch.distributed.all_gather(outs, t)
            return [int(x.item()) for x in outs]
        return [int(value)]

    def _fmt_gb(self, bytes_val: int) -> str:
        """Format bytes as GB string."""
        return f"{bytes_val / (1024**3):.2f} GB"

    def log_param_distribution_and_size(
        self,
        engine: Any,
        device: torch.device,
        rank: int,
        world_size: int
    ) -> None:
        """
        Log global param count + each GPU's share (%) and bytes of its shard.

        Args:
            engine: DeepSpeed engine
            device: Device for tensor operations
            rank: Current process rank
            world_size: Total number of processes
        """
        local_numel = self._local_shard_numel(engine, engine.module)
        local_bytes = self._local_shard_bytes(engine, engine.module)

        shard_numels = self._all_gather_i64(local_numel, device)
        shard_bytes = self._all_gather_i64(local_bytes, device)

        global_numel = sum(shard_numels)
        global_bytes = sum(shard_bytes)

        logi(
            f"[post-shard] global params: {global_numel:,}  (~{self._fmt_gb(global_bytes)} across all shards)", rank)
        for r in range(len(shard_numels)):
            pct = 100.0 * shard_numels[r] / max(global_numel, 1)
            logi(f"GPU {r+1} (rank {r}): ~{shard_numels[r]:,} params "
                 f"({pct:.2f}%), shard size ≈ {self._fmt_gb(shard_bytes[r])}", rank)

    def log_gpu_memory_snapshot(self, tag: str, rank: int) -> None:
        """
        Log per-GPU memory (global device view) + this process's PyTorch alloc/reserved.

        Args:
            tag: Label for this memory snapshot
            rank: Current process rank
        """
        if rank != 0 or not torch.cuda.is_available():
            return
        get_logger(rank).log(f"[{tag}] GPU memory snapshot (pre-data)")
        for i in range(torch.cuda.device_count()):
            # Global view from CUDA driver:
            free_b, total_b = torch.cuda.mem_get_info(i)
            used_b = total_b - free_b
            # PyTorch view (this process only):
            alloc_b = torch.cuda.memory_allocated(i)
            reserv_b = torch.cuda.memory_reserved(i)
            get_logger(rank).log(
                f"  GPU {i+1}: used { self._fmt_gb(used_b) } / total { self._fmt_gb(total_b) } | "
                f"PyTorch alloc { self._fmt_gb(alloc_b) }, reserved { self._fmt_gb(reserv_b) }"
            )

    def log_first_local_params(self, engine: Any, rank: int, max_items: int = 10) -> None:
        """
        Print the first `max_items` parameters *sharded to this rank*.

        Uses DeepSpeed's per-param ds_tensor to report the local shard size.

        Args:
            engine: DeepSpeed engine
            rank: Current process rank
            max_items: Maximum number of parameters to log
        """
        shown = 0
        for name, p in engine.module.named_parameters():
            ds_tensor = getattr(p, "ds_tensor", None)
            local_numel = int(
                ds_tensor.numel()) if ds_tensor is not None else 0
            if local_numel > 0:
                g = self.pre_global_numels.get(name, 0)
                get_logger(rank).log(
                    f"[rank {rank}] param {shown+1}: {name} | global={g:,} | "
                    f"local_shard={local_numel:,} | dtype={p.dtype} | requires_grad={p.requires_grad}"
                )
                shown += 1
                if shown >= max_items:
                    break

        if shown == 0:
            get_logger(rank).log(
                f"[rank {rank}] no local param shards found (unexpected for ZeRO-3).")
