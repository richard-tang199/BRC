# -*-coding:utf-8-*-
import argparse
from evaluation.evaluator import evaluate
from toolkit.result_plot import recon_plot, get_segments
from model.patch_soft_mask import PatchFrequencyMask
from toolkit.utils import *
import json
from dataclasses import dataclass, asdict
from toolkit.get_anomaly_score import AnomalyScoreCalculator
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='freq_new', help='Name of the model')
parser.add_argument('--dataset_name', type=str, default='TSB-AD-U', help='Name of the dataset')
parser.add_argument("--group_name", type=str, default="172", help="group in the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
parser.add_argument("--num_epochs", type=int, default=300, help="number of epochs for training")
parser.add_argument("--lr", type=float, default=2e-3, help="learning rate for training")
parser.add_argument("--eval_gap", type=int, default=20, help="training epochs between evaluations")
parser.add_argument("--use_default_config", action='store_true', help="use default configuration for training")
parser.add_argument("--patch_size", type=int, default=8, help="length of the patch")
parser.add_argument("--use_tensorboard", type=bool, default=True, help="use tensorboard for visualization")
parser.add_argument("--device", type=str, default='cuda:0', help="device for training")

if __name__ == '__main__':
    args = parser.parse_args()
    args.group_name = [args.group_name]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    save_dir = f"mask_results/{args.dataset_name}/{args.model_name}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figure", exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)

    if args.use_tensorboard:
        now = datetime.now().strftime("%m-%d-%H-%M")
        log_dir = f"logs/{args.model_name}/{args.dataset_name}/{args.dataset_name}_{args.group_name[0].zfill(3)}_{now}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(
            log_dir=log_dir)

    # args.group_name = [args.group_name]
    print(f"Start training: {args.group_name}, model_name: {args.model_name}, device: {device}")
    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name)
    use_train_data_list = train_data_list[:]
    train_subsequence_list = subsequence_list[:]

    for train_data, subsequence_length in zip(use_train_data_list, train_subsequence_list):
        train_data = train_data[:, np.newaxis]

        if len(train_data) // subsequence_length > 8:
            multiple = 8
        else:
            multiple = len(train_data) // subsequence_length
        window_length = subsequence_length * multiple
        window_length = (window_length // args.patch_size) * args.patch_size

        if len(train_data) < 1000:
            train_stride = 1
        elif 1000 <= len(train_data) < 10000:
            train_stride = 3
        elif 10000 <= len(train_data) < 60000:
            train_stride = 5
        elif 60000 <= len(train_data) < 100000:
            train_stride = 10
        else:
            train_stride = subsequence_length

        train_loader, _ = get_dataloader(
            data=train_data,
            batch_size=args.batch_size,
            window_length=window_length,
            mode="train",
            train_stride=train_stride
        )

        if args.model_name == "mix":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=window_length // args.patch_size
            )
        else:
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None
            )

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            epoch_classify_loss = 0
            epoch_reconstruction_loss = 0

            for batch_idx, (data,) in enumerate(train_loader):
                data = data.to(device)
                loss, classify_loss, reconstruction_loss = model(data, subsequence_length=subsequence_length)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_classify_loss += classify_loss.item()
                epoch_reconstruction_loss += reconstruction_loss.item()

            scheduler.step()
            print(f"Epoch {epoch}: Loss {epoch_loss / len(train_loader)}")
            mean_loss = epoch_loss / len(train_loader)
            mean_classify_loss = epoch_classify_loss / len(train_loader)
            mean_reconstruction_loss = epoch_reconstruction_loss / len(train_loader)

            if args.use_tensorboard:
                writer.add_scalar("Loss/train", mean_loss, epoch)
                writer.add_scalar("Classify_Loss/train", mean_classify_loss, epoch)
                writer.add_scalar("Reconstruction_Loss/train", mean_reconstruction_loss, epoch)

            if mean_classify_loss < 0.03 and mean_reconstruction_loss < 0.003:
                break

    for train_data, subsequence_length in zip(use_train_data_list, train_subsequence_list):
        train_data = train_data[:, np.newaxis]
        window_length = subsequence_length * multiple
        window_length = (window_length // args.patch_size) * args.patch_size
        val_loader, valid_window_converter = get_dataloader(
            data=train_data,
            batch_size=16,
            window_length=window_length,
            test_stride=subsequence_length
        )

    with torch.no_grad():
        model.eval()
        val_anomaly_score = None
        val_recon_value = None
        for batch_idx, (data,) in enumerate(val_loader):
            data = data.to(device)
            predict_score, recon_value = model.predict(data)
            if val_anomaly_score is None:
                val_anomaly_score = predict_score
            else:
                val_anomaly_score = torch.concat([val_anomaly_score, predict_score], dim=0)

            if val_recon_value is None:
                val_recon_value = recon_value
            else:
                val_recon_value = torch.concat([val_recon_value, recon_value], dim=0)

        val_anomaly_score = val_anomaly_score.detach().cpu().numpy()
        val_anomaly_score = valid_window_converter.windows_to_sequence(val_anomaly_score)
        val_recon_value = val_recon_value.detach().cpu().numpy()
        val_recon_value = valid_window_converter.windows_to_sequence(val_recon_value)

        for i, (test_data, test_label, subsequence_length) in enumerate(
                zip(test_data_list, test_label_list, subsequence_list)):
            current_group = args.group_name[i]
            test_data = test_data[:, np.newaxis]
            window_length = multiple * subsequence_length
            window_length = (window_length // args.patch_size) * args.patch_size
            test_loader, test_window_converter = get_dataloader(data=test_data,
                                                                batch_size=args.batch_size,
                                                                window_length=window_length,
                                                                test_stride=subsequence_length
                                                                )

            test_anomaly_score = None
            test_recon_value = None
            for batch_idx, (data,) in enumerate(test_loader):
                data = data.to(device)
                predict_score, recon_value = model.predict(data)
                if test_anomaly_score is None:
                    test_anomaly_score = predict_score
                else:
                    test_anomaly_score = torch.concat([test_anomaly_score, predict_score], dim=0)

                if test_recon_value is None:
                    test_recon_value = recon_value
                else:
                    test_recon_value = torch.concat([test_recon_value, recon_value], dim=0)

    test_anomaly_score = test_anomaly_score.detach().cpu().numpy()
    test_anomaly_score = test_window_converter.windows_to_sequence(test_anomaly_score)
    test_recon_value = test_recon_value.detach().cpu().numpy()
    test_recon_value = test_window_converter.windows_to_sequence(test_recon_value)

    args.group_name = args.group_name[0].zfill(3)

    save_path = os.path.join(save_dir, f"figure/group_{args.group_name}.png")
    # test_anomaly_score = np.squeeze(test_anomaly_score, axis=-1)

    anomaly_score_cal_classify = AnomalyScoreCalculator(
        mode="error",
        average_window=subsequence_length,
        add_average=False
    )

    anomaly_score_cal_recon = AnomalyScoreCalculator(
        mode="error",
        average_window=subsequence_length,
        add_average=False
    )

    recon_anomaly_score = anomaly_score_cal_recon.calculate_anomaly_score(
        raw_train_data=train_data,
        raw_test_data=test_data,
        recon_train_data=val_recon_value,
        recon_test_data=test_recon_value
    )

    classify_anomaly_score = anomaly_score_cal_classify.calculate_anomaly_score(
        raw_train_data=np.zeros_like(val_anomaly_score),
        raw_test_data=np.zeros_like(test_anomaly_score),
        recon_train_data=val_anomaly_score,
        recon_test_data=test_anomaly_score
    )

    test_recon_anomaly_score = recon_anomaly_score.test_score_all
    val_recon_anomaly_score = recon_anomaly_score.train_score_all

    test_classify_anomaly_score = classify_anomaly_score.test_score_all
    val_classify_anomaly_score = classify_anomaly_score.train_score_all

    if args.model_name == "freq_new":
        test_anomaly_score = test_recon_anomaly_score + test_classify_anomaly_score
        val_anomaly_score = val_recon_anomaly_score + val_classify_anomaly_score
    elif args.model_name == "freq_classify":
        test_anomaly_score = test_classify_anomaly_score
        val_anomaly_score = val_classify_anomaly_score
    elif args.model_name == "freq_recon":
        anomaly_score_cal_recon = AnomalyScoreCalculator(
            mode="error",
            average_window=subsequence_length,
            add_average=False
        )
        recon_anomaly_score = anomaly_score_cal_recon.calculate_anomaly_score(
            raw_train_data=train_data,
            raw_test_data=test_data,
            recon_train_data=val_recon_value,
            recon_test_data=test_recon_value
        )

        test_anomaly_score = recon_anomaly_score.test_score_all
        val_anomaly_score = recon_anomaly_score.train_score_all
    else:
        raise ValueError("Invalid model name")

    # hard masking and imputation and time domain classification

    test_anomaly_score[:test_window_converter.pad_length] = 0
    val_anomaly_score[:valid_window_converter.pad_length] = 0

    eval_result = evaluate(ground_truth=test_label,
                           anomaly_scores=test_anomaly_score,
                           subsequence=subsequence_length,
                           use_ucr=False,
                           mode="test")

    test_threshold = np.mean(test_anomaly_score) + 3 * np.std(test_anomaly_score)
    # anomaly_index = test_anomaly_score > test_threshold
    # test_anomaly_segments = get_segments(anomaly_index)

    result_path = os.path.join(save_dir, f"data/group_{args.group_name}.json")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(asdict(eval_result), f, indent=4)

    recon_plot(
        save_path=save_path,
        gap=subsequence_length,
        figure_width=10,
        figure_length=120,
        train_data=train_data_list[i],
        test_data=test_data,
        test_label=test_label,
        test_anomaly_score=test_anomaly_score,
        train_anomaly_score=val_anomaly_score,
        threshold=test_threshold,
        recon_train_data=val_recon_value,
        recon_test_data=test_recon_value
    )
