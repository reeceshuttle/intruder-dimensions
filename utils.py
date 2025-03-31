import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from collections import OrderedDict
import os
from datasets import load_dataset
from types import SimpleNamespace

# LoRAs target module names:
TARGET_MODULES_PARAM_NAMES = {
            'roberta-base': {
                'mlp_only':['intermediate.dense']+[f'{i}.output.dense' for i in range(12)],
                'qv_only':['query', 'value'],
                'attn_only':['query', 'key', 'value', 'attention.output.dense'],
                },
            'roberta-large': {
                'mlp_only':['intermediate.dense']+[f'{i}.output.dense' for i in range(24)],
                'qv_only':['query', 'value'],
                'attn_only':['query', 'key', 'value', 'attention.output.dense'],
                },
            'deberta-v2-xxlarge': {
                'mlp_only':['intermediate.dense', 'output.dense'],
                'attn_only':['query_proj', 'key_proj', 'value_proj'], # need to add output matrix
            },
        }
TARGET_MODULES_PARAM_NAMES['roberta-base']['all_linear'] = TARGET_MODULES_PARAM_NAMES['roberta-base']['mlp_only'] + TARGET_MODULES_PARAM_NAMES['roberta-base']['attn_only']
TARGET_MODULES_PARAM_NAMES['roberta-large']['all_linear'] = TARGET_MODULES_PARAM_NAMES['roberta-large']['mlp_only'] + TARGET_MODULES_PARAM_NAMES['roberta-large']['attn_only']
TARGET_MODULES_PARAM_NAMES['deberta-v2-xxlarge']['all_linear'] = TARGET_MODULES_PARAM_NAMES['deberta-v2-xxlarge']['mlp_only'] + TARGET_MODULES_PARAM_NAMES['deberta-v2-xxlarge']['attn_only']


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def print_trainable_info(model):
    print(f'---trainable parameters---')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}, {param.numel()} params')
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total trainable params: {trainable_params}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'total params: {total_params}')
    print(f'trainable percentage: {trainable_params/total_params}')
    print(f'---------------------------')

def get_finetuning_state_dict(model):
        tuned_state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                tuned_state_dict[name] = param
        return tuned_state_dict

def get_model_to_train(args, dtype=torch.float32):

    if args.task in ["winogrande", "boolq", "piqa", "sst2", "qqp"]:
        num_labels = 2
    elif args.task in ["siqa", "fever", "mnli"]:
        num_labels = 3

    if args.method == "head_only" or args.method == "full":
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=num_labels, torch_dtype=dtype)
        return model
    elif args.method in ['lora']:
        base_model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=num_labels, torch_dtype=dtype)
       
        if args.tune_layers == "all_linear":
            target_modules = TARGET_MODULES_PARAM_NAMES[args.base_model]['all_linear']
        elif args.tune_layers == "mlp_only":
            target_modules = TARGET_MODULES_PARAM_NAMES[args.base_model]['mlp_only']
        elif args.tune_layers == "attn_only":
            target_modules = TARGET_MODULES_PARAM_NAMES[args.base_model]['attn_only']
        elif args.tune_layers == "qv_only":
            target_modules = TARGET_MODULES_PARAM_NAMES[args.base_model]['qv_only']
        else:
            raise Exception(f'layers to tune {args.tune_layers} is not supported yet, maybe typo?')

        adapter_config = LoraConfig(r=args.rank, lora_alpha=args.lora_alpha, target_modules=target_modules)
        lora_model = get_peft_model(base_model, adapter_config, adapter_name=args.task)
        return lora_model

    else:
        raise Exception(f'finetuning method {args.method} is not supported yet.')


def add_label(example):
    if example['answer'] in ['ending1', 'answer1', 'option1', 'false', 'solution1']:
        label = 0
    elif example['answer'] in ['ending2', 'answer2', 'option2', 'true', 'solution2']:
        label = 1
    elif example['answer'] in ['ending3', 'answer3', 'option3']:
        label = 2
    elif example['answer'] in ['ending4', 'answer4', 'option4']:
        label = 3
    else:
        raise Exception(f'label support for task is missing.')
    return {'label': label}

def add_fever_label(example):
    if example['label'] == 'REFUTES':
        label = 0
    elif example['label'] == 'NOT ENOUGH INFO':
        label = 1
    elif example['label'] == 'SUPPORTS':
        label = 2
    else:
        raise Exception(f'label support for task is missing.')
    return {'label': label}

def get_processed_dataset(args):
    """This function should return the train, validation, and test dataloaders for the task that is specified."""

    if args.task in ["winogrande", "siqa"]:
        repo_dir = os.path.dirname(os.path.dirname(__file__))
        if args.task == "siqa":
            train_set_name = "dataset/social_i_qa/train.json"
            test_set_name = "dataset/social_i_qa/test.json"
        elif args.task == "winogrande":
            train_set_name = "dataset/winogrande/train.json"
            test_set_name = "dataset/winogrande/test.json"
        
        dataset = load_dataset('json', data_files=os.path.join(repo_dir, train_set_name))
        test_dataset = load_dataset('json', data_files=os.path.join(repo_dir, test_set_name))
        dataset['validation'] = test_dataset.pop('train') # hack, test_dataset loads the split as 'train'.

        dataset = dataset.rename_columns({"instruction":"sentence"})

        dataset = dataset.map(add_label)
        dataset = dataset.remove_columns(["input", "output", "answer"])
        for split in ['train', 'validation']:
            dataset[split] = dataset[split].add_column('idx', [i for i in range(len(dataset[split]))])

    elif args.task in ["mnli", "qqp", "sst2"]:    
        dataset = load_dataset("glue", args.task)
    elif args.task == "fever":
        dataset = load_dataset("fever", "v1.0")
        dataset = dataset.remove_columns(['evidence_annotation_id', 'evidence_id', 'evidence_wiki_url', 'evidence_sentence_id'])
        dataset = dataset.rename_columns({"claim": "sentence"})
        dataset['validation'] = dataset.pop("paper_test")
        for key in ['unlabelled_dev', 'unlabelled_test', 'labelled_dev', 'paper_dev']:
            del dataset[key]

        dataset = dataset.map(add_fever_label)        
        
    else:
        dataset = load_dataset(args.task)

    task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
    "sst2": ("sentence", None),
    "winogrande": ("sentence", None),
    "siqa": ("sentence", None),
    'fever': ("sentence", None),
    }
    sentence1_key, sentence2_key = task_to_keys[args.task]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def preprocess_function(examples):
        arguments = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*arguments, padding=False, max_length=args.max_seq_length, truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"]

        return result
    
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    return dataset


def set_trainable_params(model, args):
    # setting all to false to start:
    for name, param in model.named_parameters():
        param.requires_grad = False

    classifier_names_for_model = {
        'roberta-base':'classifier',
        'roberta-large':'classifier',
        'deberta-v2-xxlarge':'classifier',
        'gpt2-medium':'score',
        'gpt2':'score',
        'huggyllama/llama-7b':'score',
    }

    embedding_names_for_model = { # if this substring is ever in the name
        'roberta-base':['embeddings'],
        'roberta-large':['embeddings'],
        'gpt2':['wte', 'wpe'],
        'gpt2-medium':['wte', 'wpe'],
        'huggyllama/llama-7b':['embed_tokens'],
    }

    layernorm_names_for_model = {
        'roberta-base':'LayerNorm',
        'roberta-large':'LayerNorm',
        'gpt2':'.ln_',
        'gpt2-medium':'.ln_',
        'huggyllama/llama-7b':'norm',
    }

    # always doing dense tuning on the classifier layer:
    classifier_name = classifier_names_for_model[args.base_model]
    embedding_names = embedding_names_for_model[args.base_model]
    layernorm_name = layernorm_names_for_model[args.base_model]
    for name, param in model.named_parameters():
        if classifier_name in name:
            param.requires_grad = True

    if args.method == "head_only":
        return
    elif args.method == "full":
        for name, param in model.named_parameters():
            if not any([embedding_name in name for embedding_name in embedding_names]) and layernorm_name not in name and 'weight' in name:
                param.requires_grad = True
    elif args.method == 'lora':
        for name, param in model.named_parameters():
            if "weight" in name and "lora" in name: 
                param.requires_grad = True
    else:
        raise Exception(f'finetuning method {args.method} is not supported yet.')
    
    for name, param in model.named_parameters():
        if any([embedding_name in name for embedding_name in embedding_names]):
            param.requires_grad = False

def remove_peft_name_references(state_dict: OrderedDict) -> OrderedDict:
    """
    for example, a param should be named like this:
    'deberta.encoder.layer.0.attention.self.query_proj.weight'
    not like this:
    'base_model.model.deberta.encoder.layer.0.attention.self.query_proj.base_layer.weight'

    this function removes 'base_model.model.' and 'base_layer.' from the keys of the state_dict.
    """
    new_state_dict = OrderedDict()
    for param in state_dict.keys():
        new_param_name = param.replace('base_model.model.', '')
        new_param_name = new_param_name.replace('base_layer.', '')
        new_state_dict[new_param_name] = state_dict[param]
    return new_state_dict

def name_to_args(name):
    parsed_data = name.split('/')[-1].split('-')
    if parsed_data[0] == 'roberta':
        parsed_data = parsed_data[1:]
    
    args = {
        'task': parsed_data[1],
        'method': parsed_data[2],
        'max_seq_len': 512,
        'batch_size': 16,
    }

    if args['method'] == 'lora':
        args['tune_layers'] = parsed_data[5]
        args['rank'] = int(parsed_data[3].split('=')[-1])
        args['lora_alpha'] = int(parsed_data[4].split('=')[-1])

    return SimpleNamespace(**args)