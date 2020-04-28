import sys
import os
import numpy as np
import glob
from tqdm import tqdm, trange
import pickle
from  data_processors import data_processors as processors
from pprint import pprint


def get_estimate_cov(m):
    return (np.sum(m) - np.trace(m)) / np.size(m)

def get_cov_matrix(data):
    return np.cov(data)



def get_result_from_file(path):
    with open(path, 'r') as fr:
        res_str = fr.read().strip()
    return float(res_str.split(' ')[-1])


def get_logits_from_file(path):
    with open(path, 'r') as fr:
        logits_str = fr.read().strip()
        logits_list = eval(logits_str)
    return logits_list



def main():
    model_path = sys.argv[1].strip()
    if model_path[-1] == '/':
        model_path = model_path[:-1]




    if 'pkl' in model_path:
        with open(model_path, 'rb') as fr:
            results_dict = pickle.load(fr)
    else:
        model_name = model_path[model_path.rindex('models')+len('models'):]

        results_dict = {"model_name": model_name, "results": {}}


        seed_paths = glob.glob(model_path + '/seed_*/')

        seeds = [int(p[p.rindex('seed_') + len('seed_'):-1]) for p in seed_paths]

        sorted_seed_paths = sorted(zip(seed_paths, seeds), key=lambda x:x[1])

        for (s_path, seed) in tqdm(sorted_seed_paths):
            dataset_paths = glob.glob(s_path + '-*/')
            for d_path in tqdm(dataset_paths):
                dataset_name = d_path[d_path.index('/-') + len('/-'):-1]
                if dataset_name not in results_dict["results"].keys():
                    results_dict["results"][dataset_name] = {}
                #print(dataset_name)
                checkpoint_paths = glob.glob(d_path + 'checkpoint-*/')
                for c_path in checkpoint_paths:
                    checkpoint_num = int(c_path[c_path.index('checkpoint-')+len('checkpoint-'):-1])
                    if checkpoint_num not in results_dict["results"][dataset_name].keys():
                        results_dict["results"][dataset_name][checkpoint_num] = {}
                    #print(checkpoint_num)
                    # get results
                    result_path = c_path + 'eval_results.txt'
                    logits_path = c_path + 'logits_results.txt'
                    if os.path.exists(result_path):
                        eval_acc = get_result_from_file(result_path)
                    else:
                        eval_acc = None
                    if os.path.exists(logits_path):
                        eval_logits = get_logits_from_file(logits_path)
                    else:
                        eval_logits = None

                    results_dict["results"][dataset_name][checkpoint_num][seed] = {"acc": eval_acc, "logits": eval_logits}

                # results for the final checkpoint
                if "final" not in results_dict["results"][dataset_name]:
                    results_dict["results"][dataset_name]["final"] = {}
                result_path = d_path + 'eval_results.txt'
                logits_path = d_path + 'logits_results.txt'

                if os.path.exists(result_path):
                    eval_acc = get_result_from_file(result_path)
                else:
                    eval_acc = None
                if os.path.exists(logits_path):
                    eval_logits = get_logits_from_file(logits_path)
                else:
                    eval_logits = None

                results_dict["results"][dataset_name]["final"][seed] = {"acc": eval_acc, "logits": eval_logits}

        with open(model_path + '/results.pkl', 'wb') as fw:
            pickle.dump(results_dict, fw)


    print(results_dict["model_name"])
    results = results_dict["results"]
    dataset_names = list(results_dict["results"].keys())
    seeds_list = [x for x in results_dict["results"][dataset_names[0]]["final"].keys()]

    results_dict = {}
    for dataset in dataset_names:
        target_dataset_name = dataset

        #Load labels
        processor = processors[target_dataset_name]()
        label_list = processor.get_labels()
        dataset_path = './nli_data/'
        target_dataset_path = os.path.join(dataset_path, target_dataset_name.upper())
        examples = processor.get_dev_examples(target_dataset_path)

        labels = [e.label for e in examples]
        label_idxs = np.array([label_list.index(l) for l in labels])

        total_num_examples = len(examples)
        example_acc_list = None #idx -> all prediction

        target_dataset_results = results[target_dataset_name]
        for s in seeds_list:
            ckpt = "final"

            pred_logits = np.array(target_dataset_results[ckpt][s]["logits"])
            pred_label = np.argmax(pred_logits, -1)
            if target_dataset_name=='hans':
                pred_label = (pred_label!=1).astype(int)
            assert(len(pred_label) == len(label_idxs))

            example_acc = (pred_label == label_idxs)
            if example_acc_list is None:
                example_acc_list = example_acc.reshape(-1, 1)
            else:
                example_acc_list = np.concatenate([example_acc_list, example_acc.reshape(-1,1)], -1)

        final_accs = [target_dataset_results["final"][s]["acc"] * 100 for s in seeds_list]
        example_acc_list = example_acc_list.astype(float) * 100.0
        cov_matrix = get_cov_matrix(example_acc_list)

        cov_sum_estimate = get_estimate_cov(cov_matrix)

        vars = np.var(example_acc_list, axis=-1)
        ind_std = np.sqrt(np.sum(vars)) / total_num_examples

        results_dict[target_dataset_name] = {'mean': np.mean(final_accs),
                                             'std': np.std(final_accs, ddof=1),
                                             'sqrt_cov': np.sqrt(np.abs(cov_sum_estimate)),
                                             'idp_std': ind_std}
    pprint(results_dict)














if __name__ == '__main__':
    main()









