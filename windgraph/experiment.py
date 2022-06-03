import csv
import datetime
from itertools import islice
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

rel_path = Path(__file__).parent.parent


def add_exp_args(parser):
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--nb_epochs", "-e", type=int, default=10)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--slice", "-sl", type=int)
    parser.add_argument("--name", type=str)
    parser.add_argument("--config", "-c", type=str)


def run_exp(
    model, train_dl, val_dl, args, criterion=nn.MSELoss(), optimizer=torch.optim.Adam
):
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    if args.cuda:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    now = datetime.datetime.now()
    Path("logs").mkdir(exist_ok=True)
    name = f"logs/{args.name}-{now:%Y-%m-%d-%H:%M}"
    log_file = open(rel_path / (name + ".log"), "w+", 1)
    nb_parameters = sum(p.numel() for p in model.parameters())

    for p in model.parameters():
        print(p.shape, p.numel())

    print(nb_parameters)

    log_file.write(str(model) + "\n\n")
    log_file.write(f"nb_parameters {nb_parameters}\n")

    log_file.write("--------- ARGS ---------\n")
    for n in vars(args):
        log_file.write(f"{n} {getattr(args, n)}\n")
    log_file.write("------------------------\n")

    optimizer = optimizer(model.parameters(), lr=args.learning_rate)
    model.to(device)
    criterion.to(device)

    nb_train_samples, acc_train_loss = 0, 0.0
    nb_val_samples, acc_val_loss = 0, 0.0
    min_val_loss = float("inf")

    for e in range(args.nb_epochs):
        nb_train_samples, acc_train_loss = 0, 0.0

        model.train()

        if args.slice:
            pbar = tqdm(islice(iter(train_dl), args.slice), total=args.slice)
        else:
            pbar = tqdm(train_dl)

        for (cx, cy, tx), targets in pbar:
            cx = cx.to(device)
            cy = cy.to(device)
            tx = tx.to(device)
            targets = targets.to(device)
            output = model(cx, cy, tx)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_train_loss += loss.item() * cx.size(0)
            nb_train_samples += cx.size(0)
            postfix = {"train": acc_train_loss / nb_train_samples}
            pbar.set_postfix(postfix | device_dict(args.cuda))

        log_file.write(f"Train: {e+1} {acc_train_loss / nb_train_samples}\n")

        nb_val_samples, acc_val_loss = 0, 0.0
        model.eval()
        for (cx, cy, tx), targets in (pbar := tqdm(val_dl)):
            cx = cx.to(device)
            cy = cy.to(device)
            tx = tx.to(device)
            targets = targets.to(device)
            output = model(cx, cy, tx)
            loss = criterion(output, targets)

            acc_val_loss += loss.item() * cx.size(0)
            nb_val_samples += cx.size(0)
            postfix = {"val": acc_val_loss / nb_val_samples}
            pbar.set_postfix(postfix | device_dict(args.cuda))

        log_file.write(f"Val: {e+1} {acc_val_loss / nb_val_samples}\n")
        if acc_val_loss < min_val_loss:
            min_val_loss = acc_val_loss
            torch.save(model.state_dict(), rel_path / f"results/{args.name}.ckpt")
            with open(rel_path / "results/results.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
                        e,
                        args.name,
                        f"{acc_train_loss / nb_train_samples:.2f}",
                        f"{acc_val_loss / nb_val_samples:.2f}",
                    ]
                )

    return model


def device_dict(hascuda):
    if not hascuda:
        return {}

    free, tot = torch.cuda.mem_get_info(torch.cuda.current_device())
    dev_name = " ".join(torch.cuda.get_device_name().split()[-2:])
    return {dev_name: f"{(tot-free) // int(1e9)}/{tot // int(1e9)}G"}
