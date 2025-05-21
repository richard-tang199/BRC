from sched import scheduler
from dataclasses import dataclass, asdict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from evaluation.evaluator import evaluate, EvaluationResult
import os
from datetime import datetime
from toolkit.utils import *


def train_one_epoch(model, train_loader_list, optimizer, scheduler, epoch, device="cuda"):
    model.train()
    loss_list = []
    for train_loader in tqdm.tqdm(train_loader_list, desc="training dataset",
                                      leave=False, position=1, unit="dataset", dynamic_ncols=True):
        for (data, ) in train_loader:
            data = data.to(device)
            loss = model(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

    scheduler.step()
    print(f"Epoch {epoch}: {np.mean(loss_list)}")
    return np.mean(loss_list)

def test(model, test_loader, device="cuda"):
    model.eval()
    pass

def all_evaluation(model, test_loader_list, test_data_list, test_label_list, period_list, use_ucr=True, mode="test"):
    eval_result_list = []
    recon_test_list = []
    anomaly_score_list = []
    aggregate_result = EvaluationResult

    for i, test_loader in enumerate(test_loader_list):
        subsequence = period_list[i]
        recon_test = model.predict(test_loader)
        assert len(recon_test) == len(test_data_list[i])

        # anomaly score calculation
        anomaly_scores = anomaly_score_func(raw_value=test_data_list[i],
                                            predict_value=recon_test,
                                            subsequence=subsequence)
        test_label = test_label_list[i]
        eval_result = evaluate(ground_truth=test_label, anomaly_scores=anomaly_scores,
                               subsequence=subsequence,
                               use_ucr=True if use_ucr else False,
                               mode=mode)

        eval_result_list.append(eval_result)
        recon_test_list.append(recon_test)
        anomaly_score_list.append(anomaly_scores)

    if mode == "test":
        # point-wise evaluation
        aggregate_result.point_wise["auc_prc"] = np.mean([x.point_wise["auc_prc"] for x in eval_result_list])
        aggregate_result.point_wise["auc_roc"] = np.mean([x.point_wise["auc_roc"] for x in eval_result_list])
        aggregate_result.point_wise["f1_score"] = np.mean([x.point_wise["f1_score"] for x in eval_result_list])

        # SeAD evaluation
        aggregate_result.seAD["seAD_F1"] = np.mean([x.seAD["seAD_F1"] for x in eval_result_list])

        # Range-based evaluation
        aggregate_result.range_based = np.mean([x.range_based["range_F1"] for x in eval_result_list])



def all_training(model, train_loader_list, save_path, period_list,
                 test_loader_list=None, test_data_list=None, test_label_list=None,
                 optimizer=None, scheduler=None,
                 num_epoches=10, eval_freq=1):
    now = datetime.now().strftime("%m-%d-%H-%M")
    writer = SummaryWriter(os.path.join(save_path, f"logs/{now}"))

    for epoch in tqdm(range(num_epoches), position=0, desc="epochs", leave=False, unit="epoch", dynamic_ncols=True):

        # epoch_loss = model.fit_one_epoch(train_loader_list=train_loader_list)
        epoch_loss = train_one_epoch(model, train_loader_list, optimizer, scheduler, epoch)
        writer.add_scalar("loss/train", epoch_loss, epoch)

        if test_loader_list is not None:
            eval_result_list = []
            aggregate_result = EvaluationResult
            if epoch % eval_freq == 0 or epoch == num_epoches - 1:
                for i, test_loader in enumerate(test_loader_list):
                    subsequence = period_list[i]
                    recon_test = model.predict(test_loader)
                    assert len(recon_test) == len(test_data_list[i])

                    # anomaly score calculation
                    anomaly_scores = anomaly_score_func(raw_value=test_data_list[i],
                                                        predict_value=recon_test,
                                                        subsequence=subsequence)

                    test_label = test_label_list[i]
                    eval_result = evaluate(ground_truth=test_label, anomaly_scores=anomaly_scores,
                                           subsequence=subsequence,
                                           use_ucr=True if args.dataset_name == "UCR" else False,
                                           mode="test" if epoch == num_epochs - 1 else "val")
                    eval_result_list.append(eval_result)

            aggregate_result.point_wise["auc_prc"] = np.mean([x.point_wise["auc_prc"] for x in eval_result_list])
            writer.add_scalar("auc_prc/test", aggregate_result.point_wise["auc_prc"], epoch)

            if eval_result.ucr is not None:
                aggregate_result.ucr["ucr_05"] = np.mean([x.ucr["ucr_05"] for x in eval_result_list])
                writer.add_scalar("ucr_05/test", aggregate_result.ucr["ucr_05"], epoch)

            if epoch == num_epoches - 1:
                torch.save(model.state_dict(), os.path.join(save_path, f"models/model_{epoch}.pth"))
                # inference on test set
                for i, test_loader in enumerate(test_loader_list):
                    subsequence = period_list[i]
                    recon_test = model.predict(test_loader)
                    assert len(recon_test) == len(test_data_list[i])

                    # anomaly score calculation
                    anomaly_scores = anomaly_score_func(raw_value=test_data_list[i],
                                                        predict_value=recon_test,
                                                        subsequence=subsequence)
