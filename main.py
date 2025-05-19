import argparse
from toolkit.utils import *
from model.glad import Glad
from torch.utils.tensorboard import SummaryWriter
import tqdm

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="glad")
parser.add_argument("--dataset_name", type=str, default="UCR")
parser.add_argument('--task_name', type=str, default='single', help='single or all')
parser.add_argument("--group_name", type=str, default="001", help="group in the dataset")
parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model_name
    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name,
                                                                                      task_name=args.task_name)

    if args.task_name == "single":
        subsequence_list = [subsequence_list.iloc[1]]
    if args.dataset_name == "UCR":
        num_channels = 1
    else:
        raise ValueError("Dataset not supported")

    output_dir = os.path.join("output", model_name, args.dataset_name, args.task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_name == "glad":
        subsequence_length = (subsequence_list[0] // 8 + 1) * 8
        model = Glad(subsequence_length=subsequence_length, num_embeddings=30)
        model.to(device)

        train_loader_list = []
        test_loader_list = []

        for train_data, test_data in zip(train_data_list, test_data_list):
            train_loader, _ = get_dataloader(data=train_data, batch_size=args.batch_size,
                                             window_length=subsequence_length * 4, train_stride=subsequence_length // 8,
                                             mode="train")
            test_loader, _ = get_dataloader(data=test_data, batch_size=args.batch_size,
                                            window_length=subsequence_length * 4,
                                            test_stride=1, mode="test")

            train_loader_list.append(train_loader)
            test_loader_list.append(test_loader)
    else:
        raise ValueError("Model not supported")

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    for epoch in tqdm.trange(args.epochs):
        for train_loader in train_loader_list:
            results = model.fit_one_epoch(train_loader_list, lr=args.lr, device=device)
            writer.add_scalar("total_loss", results["loss"], epoch)
            writer.add_scalar("recon_loss", results["recon_loss"], epoch)
            writer.add_scalar("embed_loss", results["embedding_loss"], epoch)
            writer.add_scalar("perplexity", results["perplexity"], epoch)
