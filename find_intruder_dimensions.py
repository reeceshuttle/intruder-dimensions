import argparse
import torch
from transformers import AutoModelForCausalLM, logging
from utils import TARGET_MODULES_PARAM_NAMES, remove_peft_name_references, name_to_args, get_model_to_train

def find_number_of_intruder_dimensions(base_model, finetuned_model, model_args, threshold, k=10, verbose=False) -> int:
    base_model_state_dict = base_model.state_dict()
    test_model_state_dict = remove_peft_name_references(finetuned_model.state_dict())
    num_intruder_dimensions = 0
    for name in base_model_state_dict:
        if 'weight' in name and 'lora' not in name and any([target in name for target in TARGET_MODULES_PARAM_NAMES[model_args.base_model][model_args.tune_layers]]):
            U_base, S_base, VT_base = torch.linalg.svd(base_model_state_dict[name].float(), full_matrices=False)
            U_tuned, S_tuned, VT_tuned = torch.linalg.svd(test_model_state_dict[name].float(), full_matrices=False)
            U_dot_products = torch.abs(torch.einsum('ji,jk->ik', U_tuned, U_base))
            U_tuned_max, _ = U_dot_products.max(dim=1)
            if verbose: print(f'{name}:\n{U_tuned_max[:10]}')
            num_intruder_dimensions += len(U_tuned_max[:k][U_tuned_max[:k]<threshold])
    
    return num_intruder_dimensions

if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=300)
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=False, default="roberta-base")
    parser.add_argument("--intruder_dimension_threshold", type=float, required=False, default=0.5)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16).to(device)

    model_args = name_to_args(args.model_path)
    model_args.base_model = args.base_model
    
    finetuned_model = get_model_to_train(model_args).to(device)
    status = finetuned_model.load_state_dict(torch.load(args.model_path), strict=False)
    assert len(status.unexpected_keys) == 0, f"unexpected keys: {status.unexpected_keys}"
    if model_args.method == 'lora':
        finetuned_model.merge_and_unload()

    num_intruder_dimensions = find_number_of_intruder_dimensions(base_model, finetuned_model, model_args, 
                                                                 threshold=args.intruder_dimension_threshold, 
                                                                 verbose=args.verbose)
    print(f'number of intruder dimensions found in {args.model_path.split("/")[-1]}: {num_intruder_dimensions}')
