import torch
import argparse
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
from datetime import datetime

from utils import seed_everything, print_trainable_info, get_finetuning_state_dict, get_model_to_train, get_processed_dataset, set_trainable_params

def train_epoch(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, current_epoch, args):
    global BEST_MODEL_ACC, BEST_MODEL, BEST_MODEL_ACC_STEP, BEST_MODEL_ACC_EPOCH
    model.train()
    progress_bar = tqdm(range(len(train_dataloader.dataset)))
    optimizer.zero_grad()
    for intermediate_step, batch in enumerate(train_dataloader):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        if (intermediate_step + 1) % args.steps_per_validate == 0: # do val loss here as well.
            current_lr = max([param_group['lr'] for param_group in optimizer.param_groups])
            val_acc, val_loss = eval_dataset(model, eval_dataloader, args)
            if val_acc > BEST_MODEL_ACC:
                BEST_MODEL_ACC = val_acc
                BEST_MODEL = model.state_dict()
                BEST_MODEL_ACC_STEP = intermediate_step + current_epoch * len(train_dataloader)
                BEST_MODEL_ACC_EPOCH = current_epoch
            print(f'epoch: {current_epoch+1}, batch in epoch: {intermediate_step+1}, lr: {current_lr}, val acc: {val_acc}, val loss: {val_loss}')

        progress_bar.update(len(batch.labels))


def eval_dataset(model, eval_dataloader):
    model.eval()
    total_num_correct = 0
    total_num_samples = 0
    eval_loss = 0

    progress_bar = tqdm(range(len(eval_dataloader.dataset)))
    for step, batch in enumerate(eval_dataloader):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.item() * len(batch.labels) / len(eval_dataloader.dataset)
        
        predictions = outputs.logits.argmax(dim=-1)
        predictions = predictions.to(device)
        num_correct = torch.sum(1*(predictions == batch.labels))
        total_num_correct += num_correct
            
        total_num_samples += len(batch.labels)
        progress_bar.update(len(batch.labels))
        
    else:
        return total_num_correct/total_num_samples, eval_loss

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False, edgeitems=5, linewidth=300)

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="finetuning method name. currently supported: \'lora\', \'head_only\', \'full\', \'dora\'")
    parser.add_argument("--rank", type=int, required=False, help="rank used for lora.")
    parser.add_argument("--lora_alpha", type=int, required=False, help="alpha used in lora.")
    parser.add_argument("--task", type=str, required=True, help="task to finetune with")
    parser.add_argument("--base_model", type=str, required=True, help="base model to finetune from")
    parser.add_argument("--seed", type=int, required=False, default=42, help="seed for reproducibility")
    parser.add_argument("--batch_size", type=int, required=False, default=16, help="batch size for training")
    parser.add_argument("--num_epochs", type=int, required=False, default=10, help="number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, required=False, default=3e-4, help="learning rate for training")
    parser.add_argument("--max_seq_length", type=int, required=False, default=128, help="max sequence length for training")
    parser.add_argument("--tune_layers", type=str, required=False, default='all_linear', help="layers to tune in lora (all_linear, mlp_only, qv_only, attn_only)")    
    parser.add_argument("--weight_decay", type=float, required=False, default=0.0, help="weight decay for training")
    parser.add_argument("--save_result", type=bool, required=False, default=False, help="whether to save the results of the training")
    parser.add_argument("--steps_per_validate", type=int, required=False, default=5000, help="how many steps between validating in the training loop.")
    parser.add_argument("--details", type=str, required=False, default="")
    parser.add_argument("--save_best_acc", type=bool, required=False, default=False)

    args = parser.parse_args()
    
    assert args.task in ["mnli", "qqp", "sst2", "siqa", "winogrande", "fever"]
    if args.method not in ["head_only", "full"]: 
        assert args.rank is not None, "Rank must be specified for methods other than head_only"
        assert args.tune_layers is not None, "Must specify what to tune for methods other than head_only"

    checkpoint_dir = "trained_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%y_%H:%M")

    if args.method in ['lora']:
        checkpoint_name = f'{args.base_model}-{args.task}-{args.method}-r={args.rank}-alpha={args.lora_alpha}-{args.tune_layers}-{dt_string}-{args.details}'
    else:
        checkpoint_name = f'{args.base_model}-{args.task}-{args.method}-{dt_string}-{args.details}'

    print(args)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')

    dataset = get_processed_dataset(args)
    model = get_model_to_train(args).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    train_dataset = dataset["train"]
    if args.task == "mnli":
        val_dataset = dataset["validation_matched"]
    else:
        val_dataset = dataset["validation"]
        
    set_trainable_params(model, args)
    print_trainable_info(model)
    
    no_decay = ["bias", "LayerNorm"]
    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                     "weight_decay": args.weight_decay,},
                                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                     "weight_decay": 0.0,}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate) # add the betas here???
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.batch_size, shuffle=False, drop_last=True)
    eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    lora_paper_warmup_ratio = 0.06 # using warmup used in Hu et al
    total_training_steps = len(train_dataloader) * args.num_epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=lora_paper_warmup_ratio * total_training_steps,
        num_training_steps=total_training_steps,
    )

    BEST_MODEL_ACC = 0
    BEST_MODEL = None
    BEST_MODEL_ACC_STEP = -1
    BEST_MODEL_ACC_EPOCH = -1

    val_acc, val_loss = eval_dataset(model, eval_dataloader)
    print(f'before training: val acc: {val_acc}, val loss: {val_loss}')

    for epoch in range(args.num_epochs):
        train_epoch(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, epoch, args)
    target_checkpoint_name = checkpoint_name + '.pt'
    if args.save_result:
        if args.save_best_acc:
            tuned_state_dict = get_finetuning_state_dict(model)
            print(f'saving model with best acc...')
        else:
            tuned_state_dict = get_finetuning_state_dict(model)
        torch.save(tuned_state_dict, os.path.join(checkpoint_dir, target_checkpoint_name))
        print(f'best test acc: {BEST_MODEL_ACC} at step {BEST_MODEL_ACC_STEP+1} (in epoch {BEST_MODEL_ACC_EPOCH+1})')
        print(f'saved model to {os.path.join(checkpoint_dir, target_checkpoint_name)}')
    