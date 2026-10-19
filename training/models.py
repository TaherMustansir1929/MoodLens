from torch import nn
from torchvision import models as vision_models
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # make model non-trainable
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # use resnet3d 18 layer
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        # make model non-trainable
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_of_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_of_features, 128), nn.ReLU(), nn.Dropout(0.2)
        )

    def forward(self, x):
        # x = video frames data
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)

        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # make model non-trainable
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, x):
        x = x.squeeze(1)

        features = self.conv_layers(x)
        # features output = [batch_size, 128, 1]

        return self.projection(features.squeeze(-1))
