import argparse
from baselines.configs import *
from toolkit.training import all_training
from toolkit.utils import *


seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='cad', help='Name of the model')
parser.add_argument('--dataset_name', type=str, default='UCR', help='Name of the dataset')
parser.add_argument('--task_name', type=str, default='all', help='single or all')
parser.add_argument("--group_name", type=list, default=[1,4,8], help="group in the dataset")
parser.add_argument("--window_length", type=int, default=100, help="window length for sliding window approach")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs for training")
parser.add_argument("--lr", type=float, default=2e-5, help="learning rate for training")
parser.add_argument("--eval_gap", type=int, default=10, help="training epochs between evaluations")
parser.add_argument("--use_default_config", action='store_true', help="use default configuration for training")

if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model_name
    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name)

    if args.task_name == "single":
        subsequence_list = [subsequence_list.iloc[1]]
    if args.dataset_name == "UCR":
        num_channels = 1
    else:
        raise ValueError("Dataset not supported")

    output_dir = os.path.join("output", model_name, args.dataset_name, args.task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_name == "cad":
        from baselines.CAD.cad import MMoE

        if args.use_default_config:
            num_epochs = cad_config["num_epochs"]
            window_length = cad_config["window_length"]
        else:
            num_epochs = args.num_epochs
            window_length = args.window_length

        model = MMoE(n_multiv=num_channels, window_size=window_length - 3)
        model.to(device)

        train_loader_list = []
        test_loader_list = []

        for train_data, test_data in zip(train_data_list, test_data_list):
            train_data = train_data[:, np.newaxis]
            test_data = test_data[:, np.newaxis]

            test_data = np.concatenate(
                (np.ones([window_length - 1, num_channels]), test_data), axis=0)

            train_loader, _ = get_dataloader(data=train_data, batch_size=args.batch_size,
                                             window_length=args.window_length,
                                             mode="train")
            test_loader, _ = get_dataloader(data=test_data, batch_size=args.batch_size,
                                            window_length=args.window_length,
                                            test_stride=1, mode="test")

            train_loader_list.append(train_loader)
            test_loader_list.append(test_loader)

    else:
        raise ValueError("Model not supported")

    all_training(model_name=model_name,
                 model=model,
                 train_loader_list=train_loader_list,
                 test_loader_list=test_loader_list,
                 save_path=output_dir,
                 period_list=subsequence_list,
                 test_label_list=test_label_list,
                 train_data_list=train_data_list,
                 test_data_list=test_data_list,
                 num_epoches=args.num_epochs,
                 eval_freq=10,
                 use_ucr=True if args.dataset_name == "UCR" else False,
                 use_plot=True)
