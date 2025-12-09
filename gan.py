import torch
from typing import List, Tuple
import matplotlib.pyplot as plt
import time


class DrawingSet(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self, file_names: List[str]):
        files = [torch.load(fname) for fname in file_names]
        self.data = torch.cat(files, dim=0)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
    
    def loader(self, batch_size : int = 128, shuffle : bool = True) -> torch.utils.data.DataLoader[torch.Tensor]:
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class Generator(torch.nn.Module):
    def __init__(self, latent_dim: int, gf:int, img_shape: Tuple[int, int, int]) -> None:
        super().__init__()  # type: ignore
        self.img_shape = img_shape
        
        self.conv_l1 = torch.nn.ConvTranspose2d(in_channels=latent_dim, out_channels=4*gf, kernel_size=4, stride=1, padding=0)
        self.norm_l1 = torch.nn.BatchNorm2d(4*gf)

        self.conv_l2 = torch.nn.ConvTranspose2d(in_channels=4*gf, out_channels=2*gf, kernel_size=4, stride=2, padding=1)
        self.norm_l2 = torch.nn.BatchNorm2d(2*gf)

        self.conv_l3 = torch.nn.ConvTranspose2d(in_channels=2*gf, out_channels=gf, kernel_size=4, stride=2, padding=2)
        self.norm_l3 = torch.nn.BatchNorm2d(gf)

        self.conv_l4 = torch.nn.ConvTranspose2d(in_channels=gf, out_channels=img_shape[0], kernel_size=4, stride=2, padding=1)
        
        self.output_activation = torch.nn.Sigmoid()
        self.activation = torch.nn.ReLU()

        self.apply(init_weights)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.norm_l1(self.conv_l1(z)))
        x = self.activation(self.norm_l2(self.conv_l2(x)))
        x = self.activation(self.norm_l3(self.conv_l3(x)))
        x = self.output_activation(self.conv_l4(x))
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, gd:int, slope: float, img_shape: Tuple[int, int, int]) -> None:
        super().__init__()  # type: ignore
        
        self.conv_l1 = torch.nn.Conv2d(in_channels=img_shape[0], out_channels=gd, kernel_size=4, stride=2, padding=1)
        
        self.conv_l2 = torch.nn.Conv2d(in_channels=gd, out_channels=2*gd, kernel_size=4, stride=2, padding=1)
        self.norm_l2 = torch.nn.BatchNorm2d(2*gd)

        self.conv_l3 = torch.nn.Conv2d(in_channels=2*gd, out_channels=4*gd, kernel_size=4, stride=2, padding=1)
        self.norm_l3 = torch.nn.BatchNorm2d(4*gd)

        self.conv_l4 = torch.nn.Conv2d(in_channels=4*gd, out_channels=1, kernel_size=4, stride=2, padding=1)
        
        self.output_activation = torch.nn.Sigmoid()
        self.activation = torch.nn.LeakyReLU(slope)

        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv_l1(x))
        x = self.activation(self.norm_l2(self.conv_l2(x)))
        x = self.activation(self.norm_l3(self.conv_l3(x)))
        x = self.output_activation(self.conv_l4(x))
        return x


def init_weights(m: torch.nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # type: ignore
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # type: ignore
        torch.nn.init.constant_(m.bias.data, 0) # type: ignore


class GAN(torch.nn.Module):
    def __init__(self,
                 latent_dim: int = 32,
                 gf: int = 64,
                 gd: int = 64,
                 img_shape: Tuple[int, int, int] = (1, 28, 28),
                 slope: float = 0.2) -> None:
        super().__init__()  # type: ignore
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=self.latent_dim, gf=gf, img_shape=img_shape)
        self.discriminator = Discriminator(gd=gd, slope=slope, img_shape=img_shape)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
    
    def epoch(self,
            data_loader: torch.utils.data.DataLoader[torch.Tensor],
            optim_d:torch.optim.Optimizer,
            optim_g:torch.optim.Optimizer,
            loss_fn:torch.nn.Module) -> Tuple[float, float]:
        self.discriminator.train()
        self.generator.train()
        
        total_d_loss = 0.0
        total_g_loss = 0.0
        
        for real_images in data_loader:
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Train Discriminator
            optim_d.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1, 1, 1, device=self.device)
            output_real = self.discriminator(real_images)
            d_loss_real = loss_fn(output_real, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            fake_images = self.generator(noise)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=self.device)
            output_fake = self.discriminator(fake_images.detach())
            d_loss_fake = loss_fn(output_fake, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optim_d.step()
            
            # Train Generator
            optim_g.zero_grad()
            
            fake_labels_for_g = torch.ones(batch_size, 1, 1, 1, device=self.device)  # Trick discriminator
            output_fake_for_g = self.discriminator(fake_images)
            g_loss = loss_fn(output_fake_for_g, fake_labels_for_g)
            
            g_loss.backward()
            optim_g.step()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
        
        avg_d_loss = total_d_loss / len(data_loader)
        avg_g_loss = total_g_loss / len(data_loader)
        
        return avg_d_loss, avg_g_loss


    def training_loop(self,
                    data_loader: torch.utils.data.DataLoader[torch.Tensor],
                    num_epochs:int) -> None:
        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optim_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        loss_fn = torch.nn.BCELoss()
        test_latent = torch.randn(16, self.latent_dim, 1, 1, device=self.device)
        for epoch_idx in range(num_epochs):
            start_time = time.time()
            d_loss, g_loss = self.epoch(data_loader, optim_d, optim_g, loss_fn)
            end_time = time.time()
            self._test_batch(test_latent)
            print(f"Epoch [{epoch_idx+1}/{num_epochs}] - Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f} - Time: {end_time - start_time:.2f}s")

    def _test_batch(self, test_latent: torch.Tensor) -> None:
        test_images = self.generator(test_latent).cpu().detach().numpy()
        plt.figure(figsize=(8, 8)) # type: ignore
        for i in range(16):
            plt.subplot(4, 4, i + 1) # type: ignore
            img = test_images[i].transpose(1, 2, 0)
            if img.shape[2] == 1:
                plt.imshow(img.squeeze(), cmap='gray') # type: ignore
            else:
                plt.imshow(img) # type: ignore
            plt.axis('off') # type: ignore
        plt.tight_layout()
        plt.show() # type: ignore


if __name__ == "__main__":    # Example usage
    latent_dim = 32
    img_shape = (1, 28, 28)
    
    gan = GAN(latent_dim=latent_dim, img_shape=img_shape)
    # Assuming dataset files are available
    dataset_files = ["dog.pt", "violin.pt"]
    dataset = DrawingSet(dataset_files)
    data_loader = dataset.loader(batch_size=128, shuffle=True)
    
    gan.training_loop(data_loader, num_epochs=15)