import sys
import random
import tqdm

sys.path.append(sys.path[0] + "/model")

from layers import *


class Glad(torch.nn.Module):
    def __init__(self, subsequence_length, num_embeddings=10):
        super().__init__()
        self.subsequence_length = subsequence_length
        patch_length = self.subsequence_length // 8
        self.patcher = Patchify(sequence_length=subsequence_length,
                                patch_length=patch_length,
                                patch_stride=patch_length)
        self.proj = nn.Linear(patch_length, patch_length)
        self.subseq_encoder = nn.GRU(input_size=patch_length,
                                     hidden_size=subsequence_length,
                                     batch_first=True,
                                     bidirectional=True,
                                     num_layers=1)
        self.vector_quantizer = VectorQuantizer(n_e=num_embeddings, e_dim=subsequence_length, beta=0.25)
        self.revesr_proj = nn.Linear(patch_length, patch_length)
        self.current_epoch = 0

    def forward(self, x: torch.Tensor):
        # x: (batch_size, window_size, num_channels)
        x = x.permute(0, 2, 1)  # (batch_size, num_channels, window_size)
        # x: (batch_size, num_channels, num_subsequences, subsequence_length)
        x = x.unfold(-1, size=self.subsequence_length, step=self.subsequence_length)
        # x_patched: (batch_size, num_channels, num_subsequences, num_patches, patch_length)
        x_patched = self.patcher(x)
        num_patches = x_patched.shape[-2]
        # x_encoding: (batch_size, num_channels, num_subsequences, num_patches, patch_length)
        patch_encoding = self.proj(x_patched)
        # subsequence_encoding: (batch_size * num_channels * num_subsequences, subsequence_length)
        patch_encoding = patch_encoding.reshape(-1, num_patches, self.patcher.patch_length)
        _, subsequence_encoding = self.subseq_encoder(patch_encoding)
        subsequence_encoding = subsequence_encoding.mean(dim=0).reshape(x.shape[0], x.shape[1], -1,
                                                                        self.subsequence_length)
        # quantized_encoding: (batch_size, num_channels, num_subsequences, subsequence_length)
        # index: (batch_size, num_channels, num_subsequences)
        embedding_loss, quantized_encoding, perplexity, index = self.vector_quantizer(subsequence_encoding)
        # x_revesr: (batch_size, num_channels, num_subsequences, num_patches, patch_length)
        x_revesr = quantized_encoding.reshape(x.shape[0], x.shape[1], x.shape[2], num_patches, -1)
        x_recon = self.revesr_proj(x_revesr)
        # calculate reconstruction loss
        recon_loss = nn.MSELoss(reduction='mean')(x_recon, x_patched)
        # calculate total loss
        loss = 0.2 * embedding_loss + recon_loss
        return {
            "all_loss": loss,
            "embedding_loss": embedding_loss,
            "recon_loss": recon_loss,
            "perplexity": perplexity,
            "recons": x_recon
        }

    def fit_one_epoch(self, train_loader_list, lr=1e-3, device="cuda:0"):
        self.current_epoch += 1
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        random.shuffle(train_loader_list)

        loss_list = []
        recon_loss_list = []
        embedding_loss_list = []
        perplexity_list = []

        for train_loader in tqdm.tqdm(train_loader_list, desc="training dataset",
                                      leave=False, position=1, unit="dataset", dynamic_ncols=True):
            for (data,) in train_loader:
                data = data.to(device)
                result = self.forward(data)
                loss = result["all_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(result["all_loss"].item())
                recon_loss_list.append(result["recon_loss"].item())
                embedding_loss_list.append(result["embedding_loss"].item())
                perplexity_list.append(result["perplexity"].item())

        scheduler.step()
        print(f"Epoch {self.current_epoch}: {sum(loss_list) / len(loss_list)}")

        return {
            "loss": sum(loss_list) / len(loss_list),
            "recon_loss": sum(recon_loss_list) / len(recon_loss_list),
            "embedding_loss": sum(embedding_loss_list) / len(embedding_loss_list),
            "perplexity": sum(perplexity_list) / len(perplexity_list)
        }

    # def predict(self,


if __name__ == '__main__':
    model = Glad(subsequence_length=32)
    x = torch.rand(16, 64, 1)
    out = model(x)
    for k, v in out.items():
        print(k, v)
