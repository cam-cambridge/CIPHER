import os
import torch
from torch.optim import Adam
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
from matplotlib import pyplot as plt
import wandb
from tqdm import tqdm
from huggingface_hub import hf_hub_download

class VisionExpert:
    def __init__(self, load_dir=None):
        """
        Initializes the VisionExpert class by setting the model and processor.
        
        Parameters:
        model_name (str): The pretrained model name. Ignored if loading from `load_dir`.
        load_dir (str): Directory to load a previously saved model and processor from.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vexpert = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT
        )
        num_features = self.vexpert.fc.in_features
        self.vexpert.fc = nn.Sequential(
            nn.Linear(num_features, 1),
        )
        self.vexpert.to(self.device)

        for param in self.vexpert.parameters():
            param.requires_grad = True

        self.opt = Adam(self.vexpert.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10000, gamma=0.5)
        self.criterion = nn.MSELoss()

        if load_dir:
            self.load(load_dir)

        self.steps=0

    def count_step(self):
        self.steps+=1
        # print(f"\n{self.steps}\n")

    def batch_loader(self, inputs):
        batch = {"img": [], "y": []}

        for input_row in inputs:
            try:
                # Load the image
                img = input_row["image"]  # Load as a color image
                # img.save(f"testing/{self.steps}.png")

                if img is None:
                    raise FileNotFoundError(f"Image not found at path: {img_path}")
                
                transform = transforms.Compose([
                    # transforms.RandomAffine(degrees=15, scale=(0.8,1.2), translate=(0, 0.1)),
                    transforms.CenterCrop(224),
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])    
                    ])

                img= transform(img)

                batch["img"].append(img)

            except Exception as e:
                print(f"Error loading image: {e}")
                batch["img"].append(None)

            # Append the label (flow_rate) to the batch
            batch["y"].append(input_row["flow_rate"])
        
        # Convert lists to tensors
        batch["img"] = torch.stack(batch["img"]).to(self.device)
        batch["y"] = torch.tensor(batch["y"]).to(self.device)
        batch["y"] = torch.log(batch["y"]/100).to(self.device)
        
        return batch    

    def active_learning(self, input):
        self.vexpert.train()
        batch = self.batch_loader(input)

        self.opt.zero_grad()
        y_hat = self.vexpert(batch['img']).squeeze()
        loss = self.criterion(y_hat, batch['y'])
        loss.backward()
        
        self.opt.step()
        self.scheduler.step()
        self.count_step()

        return y_hat.detach(), batch['y'], list(map(int, (torch.exp(y_hat.detach()) * 100).detach().cpu().tolist()))

    def validate_step(self, input):

        self.vexpert.eval()
        with torch.no_grad():
            batch= self.batch_loader(input)
            y_hat = self.vexpert(batch['img']).squeeze()

            predictions_mapped= torch.exp(y_hat)*100
            ground_truths_mapped= torch.exp(batch['y'])*100
            avg_val_metric= torch.mean(torch.abs(ground_truths_mapped - predictions_mapped).detach()).item()

        return y_hat.detach(), batch['y'], list(map(int, (torch.exp(y_hat.detach()) * 100).detach().cpu().tolist()))


    def validate_model(self, val_batches):
       
        self.vexpert.eval()  # Set the model to evaluation mode
        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():  # Disable gradient computation
            for i, batch in enumerate(val_batches):
                batch_data = self.batch_loader(batch)
                y_hat = self.vexpert(batch_data['img']).squeeze()
                all_predictions.extend(y_hat.cpu().tolist())
                all_ground_truths.extend((batch_data['y']).cpu().tolist())

        return torch.tensor(all_predictions), torch.tensor(all_ground_truths)

    def create_batches(self, dataset, batch_size):
        """
        Split the dataset into batches of specified size.
        
        Args:
            dataset (pd.DataFrame): The dataset to batch.
            batch_size (int): The size of each batch.
        
        Returns:
            list: A list of batches, each batch is a DataFrame.
        """
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # Calculate total number of batches
        return [dataset[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    def validate_internally(self, samples):

        # Example usage
        batch_size = 4  # Define your batch size
        val_batches = self.create_batches(samples, batch_size)

        predictions, ground_truths = self.validate_model(val_batches)
        predictions_mapped= torch.exp(predictions)*100
        ground_truths_mapped= torch.exp(ground_truths)*100
        avg_val_metric= torch.mean(torch.abs(ground_truths_mapped - predictions_mapped).detach()).item()
        # wandb.log({"expert_MAE_loss (val)": avg_val_metric}, step=self.steps)

        # # Plot predictions vs ground truth
        # plt.figure(figsize=(8, 8))
        # plt.scatter(ground_truths_mapped.detach().cpu().numpy(), predictions_mapped.detach().cpu().numpy(), alpha=0.5, label="Predictions vs Ground Truth")
        # plt.plot([min(ground_truths_mapped.detach().cpu().numpy()), max(ground_truths_mapped.detach().cpu().numpy())], [min(ground_truths_mapped.detach().cpu().numpy()), max(ground_truths_mapped.detach().cpu().numpy())], 'r--', label="Ideal Fit")
        # plt.xlabel("Ground Truth")
        # plt.ylabel("Predictions")
        # plt.title(f"Predictions vs Ground Truth (Epoch {1})")
        # plt.legend()
        # plt.grid(True)
        
        # plt.ylim([30,300])
        # plt.xlim([30,300])
        
        # # Save the plot
        # plot_path = f"pred_vs_gt_internal_validation.png"
        # plt.savefig(plot_path)
        # print(f"Plot saved at {plot_path}")
        # plt.close()

    def warmup(self, warmup_dataset, validation_dataset, warmup_steps=1000):


        # Example usage
        batch_size = 4  # Define your batch size
        warmup_batches = self.create_batches(warmup_dataset, batch_size)
        val_batches = self.create_batches(validation_dataset, batch_size)

        # Training loop
        for epoch in tqdm(range(1), desc="Epoch Progress"):
            with tqdm(total=len(warmup_batches), desc=f"Training Progress (Epoch {epoch})", leave=False) as batch_bar:
                    for i,batch in enumerate(warmup_batches):
                        _, _, _ = self.active_learning(batch)
                        batch_bar.update(1)

                        if i%5000==0:

                            # Validation step
                            predictions, ground_truths = self.validate_model(val_batches)
                            predictions_mapped= torch.exp(predictions)*100
                            ground_truths_mapped= torch.exp(ground_truths)*100
                            avg_val_metric= torch.mean(torch.abs(ground_truths_mapped - predictions_mapped).detach()).item()
                            wandb.log({"expert_MAE_loss (val)": avg_val_metric}, step=self.steps)
                            
                            # # Plot predictions vs ground truth
                            # plt.figure(figsize=(8, 8))
                            # plt.scatter(ground_truths_mapped.detach().cpu().numpy(), predictions_mapped.detach().cpu().numpy(), alpha=0.5, label="Predictions vs Ground Truth")
                            # plt.plot([min(ground_truths_mapped.detach().cpu().numpy()), max(ground_truths_mapped.detach().cpu().numpy())], [min(ground_truths_mapped.detach().cpu().numpy()), max(ground_truths_mapped.detach().cpu().numpy())], 'r--', label="Ideal Fit")
                            # plt.xlabel("Ground Truth")
                            # plt.ylabel("Predictions")
                            # plt.title(f"Predictions vs Ground Truth (Epoch {epoch})")
                            # plt.legend()
                            # plt.grid(True)
                            
                            # plt.ylim([30,300])
                            # plt.xlim([30,300])
                            
                            # idx=i*16*20

                            # plot_path = f"pred_vs_gt_epoch_{idx}.png"
                            # plt.savefig(plot_path)
                            # print(f"Plot saved at {plot_path}")
                            # plt.close()

                            # self.save("/home/cm2161/rds/hpc-work")

    def save(self, save_dir):
        """
        Save the model state dictionary and optimizer state.
        
        Args:
            save_dir (str): Directory where the model will be saved.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save({
            'model_state_dict': self.vexpert.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'steps': self.steps
        }, os.path.join(save_dir, "vexpert_checkpoint.pth"))
        print(f"Model saved to {save_dir}")

    
    def load(self, load_dir):
        """
        Load the model state dictionary and optimizer state.
        Downloads from HuggingFace if not found locally.
        
        Args:
            load_dir (str): Directory where the model checkpoint is saved.
        """
        
        checkpoint_path = f"{load_dir}/vexpert_checkpoint.pth"
        
        # If checkpoint doesn't exist locally, download from HuggingFace
        if not os.path.exists(checkpoint_path):
            print(f"Expert checkpoint not found at {checkpoint_path}")
            print("Downloading from HuggingFace: cemag/cipher_printing/vexpert_checkpoint.pth")
            
            checkpoint_path = hf_hub_download(
                repo_id='cemag/cipher_printing',
                filename='vexpert_checkpoint.pth',
                cache_dir=load_dir,
            )
            print(f"Downloaded checkpoint to: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.vexpert.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.steps = checkpoint['steps']
        print(f"Model loaded from {checkpoint_path}")

# wandb.init(project="Pr-Intern", name="vexpert_isolated")
# model = VisionExpert(load_dir="/home/cm2161/rds/hpc-work/")
