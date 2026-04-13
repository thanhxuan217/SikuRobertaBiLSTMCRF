# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 17:27
"""
File này định nghĩa lớp `Train_single` kế thừa từ `CMD` in `cmd_single.py`. 
Đảm nhiệm vòng lặp huấn luyện (train loop) qua các epochs dành cho mô hình Single-CRF, 
bao gồm việc nạp dữ liệu từ loader của một task duy nhất, 
thiết lập phân cấp learning rate (classification head học nhanh hơn) và cấu hình optimizer.
Hỗ trợ QLoRA: khi bật --use_qlora, dùng paged_adamw_8bit và chỉ optimize LoRA + head params.
"""

from datetime import datetime, timedelta
import os
import torch

from .cmd_single import CMD

from ..utils.load_streaming import Load
from ..utils.metric import Metric

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader


class Train_single(CMD):
    """
    Lớp Trainer dành cho Mô hình Single-CRF
    (Sử dụng roberta_bilstm_crf thay vì bigram_bert_model).
    Hỗ trợ QLoRA fine-tuning khi args.use_qlora = True.
    """

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--data', default='data/ptb_pos',
                               help='path to train file')
        return subparser

    def _move_qlora_to_gpu(self, device):
        """
        Đảm bảo toàn bộ model params nằm trên GPU cho QLoRA.
        BERT quantized weights đã trên GPU nhờ device_map="auto".
        Cần move: char_embed, lstm, mlp, crf, projection, LoRA adapters.
        """
        # 1. Move các module KHÔNG phải BERT (char_embed, lstm, mlp, crf, dropout...)
        for name, module in self.model.named_children():
            if name != 'feat_embed':
                module.to(device)
        
        # 2. Move projection layer trong feat_embed (nếu có)
        if hasattr(self.model.feat_embed, 'projection'):
            self.model.feat_embed.projection.to(device)
        
        # 3. Move LoRA adapters — .to() trên PEFT model chỉ move float params, không chạm quantized weights
        self.model.feat_embed.bert.to(device)
        
        # 4. Quét lại: nếu còn trainable param nào trên CPU thì ép move
        #    (Bắt các trường hợp device_map="auto" hoặc PEFT init để sót)
        moved_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and not param.is_cuda:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
                moved_count += 1
                print(f"  [QLoRA] Force-moved '{name}' ({param.shape}) -> {device}")
        
        # 5. Quét buffers (batch_norm running_mean, v.v.) - cũng cần trên cùng device
        for name, buf in self.model.named_buffers():
            if not buf.is_cuda:
                # Chỉ move buffer của các module không phải BERT quantized
                if 'feat_embed.bert' not in name:
                    buf.data = buf.data.to(device)
        
        if moved_count > 0:
            print(f"  [QLoRA] Đã force-move {moved_count} params lên {device}")
        else:
            print(f"  [QLoRA] Tất cả trainable params đã trên {device}")

    def _move_optimizer_to_gpu(self, device):
        """
        Move tất cả optimizer states lên GPU.
        Cần thiết sau khi load checkpoint với map_location='cpu'.
        """
        moved = 0
        for state in self.optimizer.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor) and not val.is_cuda:
                    state[key] = val.to(device)
                    moved += 1
        if moved > 0:
            print(f"  [QLoRA] Moved {moved} optimizer state tensors -> {device}")

    def __call__(self, args):
        super(Train_single, self).__call__(args)
        print('Preprocess the data')
        loader = Load(args) # Gọi đối tượng tiền xử lý file từ "utils/load_single.py"
        
        # Collate (gom batch) version single model
        collate_fn = loader.collate_fn_crf_last
        
        print(f"{args}")
        train, dev = loader.train, loader.validation
        # print(len(train))
        self.trainset = DataLoader(train, batch_size=args.batch_size,
                                   collate_fn=collate_fn)
        self.devset = DataLoader(dev, batch_size=args.batch_size,
                                 collate_fn=collate_fn)

        use_qlora = getattr(args, 'use_qlora', False)

        # QLoRA bắt buộc phải có CUDA/GPU
        if use_qlora and not torch.cuda.is_available():
            raise RuntimeError(
                "QLoRA yêu cầu GPU (CUDA). Không phát hiện GPU khả dụng.\n"
                "Hãy chạy lại mà không dùng --use_qlora, hoặc kiểm tra CUDA_VISIBLE_DEVICES."
            )

        self.model = self.model_cl(args)
        if use_qlora:
            self._move_qlora_to_gpu(args.device)
        else:
            self.model = self.model.to(args.device)

        print("Create the model.")
        print(f"{self.model}")

        
        # Setup Learning Rate
        lr = args.bert_lr
        decay = args.bert_decay

        if use_qlora:
            # === QLoRA Mode ===
            # BERT backbone đã frozen & quantized, chỉ optimize:
            # 1. LoRA adapter params (trong BERT)
            # 2. BiLSTM, char_embed, projection params
            # 3. MLP + CRF head params (learning rate cao hơn)
            import bitsandbytes as bnb
            
            # Thu thập LoRA params (trainable params trong BERT)
            lora_params = [p for p in self.model.feat_embed.bert.parameters() if p.requires_grad]
            
            # Thu thập non-BERT trainable params (char_embed, projection, lstm)
            non_head_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'feat_embed.bert' in name:
                    continue  # Đã xử lý ở trên
                if 'crf' in name or 'mlp' in name:
                    continue  # Xử lý riêng bên dưới
                non_head_params.append(param)
            
            # Classification head params (MLP + CRF) - LR cao hơn
            times, weight_decay = 10, 0.01
            head_params = [
                {'params': self.model.crf.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
                {'params': self.model.mlp.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
            ]
            
            param_groups = [
                {'params': lora_params, 'lr': lr},
                {'params': non_head_params, 'lr': lr},
            ] + head_params
            
            # Dùng Paged AdamW 8-bit để tiết kiệm VRAM cho optimizer states
            self.optimizer = bnb.optim.PagedAdamW8bit(
                param_groups,
                lr=lr,
                betas=(args.mu, args.nu),
                eps=args.epsilon,
            )
            
            # In thông tin trainable params
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[QLoRA] Total params: {total_params:,} | Trainable: {trainable_params:,} "
                  f"({100 * trainable_params / total_params:.2f}%)")
        else:
            # === Normal Mode (không QLoRA) ===
            # Chia thành 2 nhóm thông số: 1 là của cốt lõi BERT/LSTM, 2 là mô-đun MLP/CRF phía trên cùng
            bert_lstm_params = [param for name, param in self.model.named_parameters()
                                if 'crf' not in name and 'mlp' not in name]
                                
            # Đầu não phân loại (Classification Head) được cấp quyền học nhanh gấp 10 lần (`times=10`)
            times, weight_decay = 10, 0.01
            crf_params = [
                {'params': self.model.crf.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
                {'params': self.model.mlp.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
            ]

            # Optimizer dùng Adam
            self.optimizer = Adam([{'params': bert_lstm_params}] + crf_params,
                                  lr,
                                  (args.mu, args.nu),
                                  args.epsilon)

        decay_steps = args.decay_epochs * len(self.trainset)
        self.scheduler = ExponentialLR(self.optimizer,
                                       decay ** (1 / decay_steps))
        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        saved_best_model = False
        start_step = 0
        best_step = 0

        if getattr(args, 'resume', False) and os.path.exists(args.save_model):
            print(f"Resuming from checkpoint {args.save_model}...")
            from parsering.checkpoint import load_checkpoint
            try:
                from safetensors.torch import load_file as safe_load_file
            except ImportError:
                safe_load_file = None
                
            checkpoint = load_checkpoint(args.save_model, map_location='cpu')
            
            state_dict = checkpoint['state_dict']
            if use_qlora:
                base_name = os.path.splitext(os.path.basename(args.save_model))[0]
                lora_dir = os.path.join(os.path.dirname(args.save_model), f'lora_adapters_{base_name}')
                if not os.path.exists(lora_dir):
                    old_lora_dir = os.path.join(os.path.dirname(args.save_model), 'lora_adapters')
                    if os.path.exists(old_lora_dir):
                        lora_dir = old_lora_dir
                if os.path.exists(lora_dir):
                    print(f"Loading LoRA adapters from {lora_dir}...")
                    from peft import set_peft_model_state_dict
                    sf_path = os.path.join(lora_dir, "adapter_model.safetensors")
                    bin_path = os.path.join(lora_dir, "adapter_model.bin")
                    if os.path.exists(sf_path) and safe_load_file:
                        lora_weights = safe_load_file(sf_path)
                    elif os.path.exists(bin_path):
                        lora_weights = torch.load(bin_path, map_location='cpu', weights_only=True)
                    else:
                        raise FileNotFoundError(f"Cannot find adapter models in {lora_dir}")
                    set_peft_model_state_dict(self.model.feat_embed.bert, lora_weights)
            self.model.load_state_dict(state_dict, strict=False)
            
            # Sau khi load state_dict từ CPU, đảm bảo model params lại trên GPU
            if use_qlora:
                self._move_qlora_to_gpu(args.device)
            
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # Move optimizer states lên GPU (checkpoint load với map_location='cpu')
                if use_qlora:
                    self._move_optimizer_to_gpu(args.device)
                print("Restored optimizer state.")
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                print("Restored scheduler state.")
            if 'best_metric' in checkpoint:
                best_metric = checkpoint['best_metric']
            if 'best_e' in checkpoint:
                best_step = checkpoint['best_e']
            start_step = checkpoint.get('step', 0)
            print(f"Resuming training starting from global step {start_step}")

        def infinite_loader(dataloader):
            while True:
                for batch in dataloader:
                    yield batch

        train_iter = infinite_loader(self.trainset)
        
        if start_step > 0:
            skip_batches = start_step % len(self.trainset)
            if skip_batches > 0:
                print(f"Fast-forwarding iterator by {skip_batches} batches...")
                for _ in range(skip_batches):
                    next(train_iter)
                    
        global_step = start_step
        max_steps = getattr(args, 'max_steps', 100000)
        eval_steps = getattr(args, 'eval_steps', 5000)

        while global_step < max_steps:
            steps_to_run = min(eval_steps, max_steps - global_step)
            start = datetime.now()
            print(f"Training steps {global_step + 1} to {global_step + steps_to_run} / {max_steps}:")

            train_stats = self.train(train_iter, steps_to_run=steps_to_run, global_step=global_step, best_metric=best_metric, best_e=best_step)
            global_step += steps_to_run

            print(
                f"{'train:':6} Loss: {train_stats['avg_loss']:.4f} | "
                f"Time: {timedelta(seconds=int(train_stats['elapsed_seconds']))}"
            )

            loss, metric_punc = self.evaluate(self.devset)
            print(f"{'dev:':6} Loss: {loss:.4f}")
            print('punc', metric_punc)

            t = datetime.now() - start
            best_model_path = args.save_model.replace('.pth', '_best.pth') if args.save_model.endswith('.pth') else args.save_model + '_best'
            # save the model if it is the best so far
            if metric_punc > best_metric:
                best_step, best_metric = global_step, metric_punc
                self.model.save(best_model_path, optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), epoch=0, step=global_step, best_metric=best_metric, best_e=best_step)
                saved_best_model = True
                print(f"{t} elapsed (saved best metric to {best_model_path})\n")
            else:
                print(f"{t} elapsed\n")
                
            # Unconditionally save latest state
            self.model.save(args.save_model, optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), epoch=0, step=global_step, best_metric=best_metric, best_e=best_step)
            print(f"(saved current state to {args.save_model})\n")
            total_time += t

            completed_ratio = global_step / max_steps * 100
            print(
                f"[progress] step {global_step}/{max_steps} ({completed_ratio:.1f}%) | "
                f"total_elapsed={total_time}\n"
            )

            patience_intervals = getattr(args, 'patience', 10)
            patience_steps = patience_intervals * eval_steps
            if global_step - best_step >= patience_steps:
                print(f"Early stopping triggered! No improvement for {patience_steps} steps.")
                break

        if not saved_best_model:
            best_model_path = args.save_model.replace('.pth', '_best.pth') if args.save_model.endswith('.pth') else args.save_model + '_best'
            print(
                f"No best checkpoint was saved during training. "
                f"Saving the final model to {best_model_path}."
            )
            try:
                self.model.save(best_model_path, optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), epoch=0, step=global_step, best_metric=best_metric, best_e=best_step)
            except UnboundLocalError:
                self.model.save(best_model_path)
            saved_best_model = True

        best_model_path = args.save_model.replace('.pth', '_best.pth') if args.save_model.endswith('.pth') else args.save_model + '_best'
        if os.path.exists(best_model_path):
            print(f"Loading best checkpoint for final evaluation: {best_model_path}")
            self.model = self.model_cl.load(best_model_path)
        elif os.path.exists(args.save_model):
            print(f"Loading latest checkpoint for final evaluation: {args.save_model}")
            self.model = self.model_cl.load(args.save_model)
        else:
            print(
                f"Warning: neither '{best_model_path}' nor '{args.save_model}' was found after training. "
                f"Using the in-memory model for final evaluation."
            )
        loss, metric_punc = self.evaluate(self.devset)

        print(f"max score of dev is {best_metric.score:.2%} at step {best_step}")
        print(f"the loss of dev at step {best_step} is {loss:.2f}")

        print(f"Total time elapsed: {total_time}s")
