import time, sys
import numpy as np
import argparse
import os 
import wandb

import json
import torch

from classifier import Classifier


def get_wandb_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    print("Loaded config:", config)
    return config


def launch_sweep():
    run = wandb.init()
    lr  =  wandb.config.LEARNING_RATE
    epochs = wandb.config.EPOCHS
    main(lr, epochs)


def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)



def load_label_output(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        return [line.strip().split("\t")[0] for line in f if line.strip()]



def eval_list(glabels, slabels):
    if (len(glabels) != len(slabels)):
        print("\nWARNING: label count in system output (%d) is different from gold label count (%d)\n" % (
        len(slabels), len(glabels)))
    n = min(len(slabels), len(glabels))
    incorrect_count = 0
    for i in range(n):
        if slabels[i] != glabels[i]: incorrect_count += 1
    acc = (n - incorrect_count) / n
    return acc*100



def train_and_eval(classifier, trainfile, devfile, testfile, device, lr, epochs):
    print("  %s.1. Training the classifier...")
    classifier.train(trainfile, devfile, device, lr=lr, epochs=epochs)
    print("  %s.2. Eval on the dev set...", end="")
    slabels = classifier.predict(devfile, device)
    glabels = load_label_output(devfile)
    devacc = eval_list(glabels, slabels)
    print(" Acc.: %.2f" % devacc)
    testacc = -1
    if testfile is not None:
        # Evaluation on the test data
        print("  %s.3. Eval on the test set...", end="")
        slabels = classifier.predict(testfile)
        glabels = load_label_output(testfile)
        testacc = eval_list(glabels, slabels)
        print(" Acc.: %.2f" % testacc)
    return (devacc, testacc)


def main(lr, epochs):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--n_runs', help='Number of runs.', type=int, default=5)
    argparser.add_argument('-g', '--gpu', help='GPU device id on which to run the model', type=int)
    args = argparser.parse_args()
    device_name = "cpu" if args.gpu is None else f"cuda:{args.gpu}"
    device = torch.device(device_name)
    n_runs = args.n_runs
    set_reproducible()
    datadir = "data/"
    trainfile =  datadir + "traindata.csv"
    devfile =  datadir + "devdata.csv"
    testfile = None
    # testfile = datadir + "testdata.csv"

    # Runs
    start_time = time.perf_counter()
    devaccs = []
    testaccs = []
    classifier =  Classifier()
    devacc, testacc = train_and_eval(classifier, trainfile, devfile, testfile, device, lr=lr, epochs=epochs)
    devaccs.append(np.round(devacc,2))
    testaccs.append(np.round(testacc,2))
    total_exec_time = (time.perf_counter() - start_time)
    wandb.log(
        {
            "predicted_dev_acc": devacc, 
            "predicted_test_acc": testacc,
            "predicted_mean_dev_acc": np.mean(devaccs),
            "predicted_mean_test_acc": np.mean(testaccs),
        }
    )
    print("Dev accs:", devaccs)
    print("Test accs:", testaccs)

    print("Mean Dev Acc.: %.2f (%.2f)" % (np.mean(devaccs), np.std(devaccs)))
    print("Mean Test Acc.: %.2f (%.2f)" % (np.mean(testaccs), np.std(testaccs)))
    print("\nExec time: %.2f s. ( %d per run )" % (total_exec_time, total_exec_time / n_runs))


SWEEP = True
if __name__ == "__main__":
    if SWEEP:
        sweep_config = get_wandb_config('config/sweep_config.json')
        sweep_id = wandb.sweep( sweep=sweep_config, project='nlp-sa')
        wandb.agent(sweep_id, function=launch_sweep, count=50)
    else:
        lr = 1e-5
        epochs = 2
        run = wandb.init(
            project="nlp-sa",
            config={
                "lr": lr,
                "epochs": epochs
            }
        )
        main(lr=lr, epochs=epochs)



