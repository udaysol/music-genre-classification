import engine, utils, models, data_setup
import torch
import pandas as pd

from torch import nn
from torchmetrics import Accuracy
from pathlib import Path
from torchinfo import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Creating CSV for filepaths and labels from the data
data_dir = Path("Data/genres_original")
df, class_names = data_setup.get_csv(data_path=data_dir)

# Creating Model and transforms
resnet34, transforms = models.create_resnet34(num_classes=len(class_names),
                                              seed=42,
                                              device=device)

# ---Uncomment to get model info.---
# summary(resnet34, input_size=[32, 3, 288, 288],
#         col_names=['input_size', 'output_size', 'trainable'],
#         col_width=20,
#         row_settings=['var_names'])

# Creating Dataset and dataloaders
data_df = pd.read_csv("dataset.csv")
audio_dataset = data_setup.custom_dataset(dataframe=data_df,
                                       transform=transforms)
BATCH_SIZE = 32
TEST_SIZE = 0.2     # 0.2 = 20% -> 80-20 split

train_dataloader, test_dataloader = data_setup.create_dataloaders(dataset=audio_dataset,
                                                                    batch_size=BATCH_SIZE,
                                                                    test_size=TEST_SIZE)

# Loss function
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer
optimizer = torch.optim.AdamW(params=resnet34.parameters(),
                              lr=1e-4,
                              weight_decay=1e-4)
# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=2)

# Accuracy function
accuracy_fn = Accuracy(task='multiclass',
                       num_classes=len(class_names)).to(device)

# Training The model
utils.set_seed(42)
NUM_EPOCHS = 5

results = engine.train_model(model=resnet34,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             scheduler_fn=scheduler,
                             accuracy_fn=accuracy_fn,
                             epochs=NUM_EPOCHS,
                             device=device)

# Saving the Model 
save_dir = Path('saved_models')
utils.save_model(model=resnet34,
                 target_dir=save_dir,
                 model_name="music_gnere_resnet34_10_epochs.pth")