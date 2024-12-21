import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        
        # Main convolution path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add shortcut connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ResNet1D(nn.Module):
    def __init__(self, in_channels = 8, num_classes = 1):
        super(ResNet1D, self).__init__()
       
        # Initial Convolutional layer
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
       
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 3)  # 3 residual blocks
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
       
        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)
   
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        # First block handles stride and potential channel change
        layers.append(BasicBlock1D(in_channels, out_channels, stride))
        
        # Subsequent blocks maintain the same channel and spatial dimensions
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
       
        return nn.Sequential(*layers)
   
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
       
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
       
        # Fully connected output layer
        x = self.fc(x)
       
        return x


class Conv1D_v2(nn.Module):
    def __init__(self, channels = 8):
        super(Conv1D_v2, self).__init__() 

        self.seq = nn.Sequential(
         
         nn.Conv1d(channels, channels * 2, kernel_size= 3),
         nn.ReLU(),
        #  nn.Dropout1d(0.2),
        nn.BatchNorm1d(num_features=channels * 2),
         nn.MaxPool1d(kernel_size= 2), 

        nn.Conv1d( channels* 2, channels * 4, kernel_size= 3),
        nn.ReLU(),
        
        #  nn.Dropout1d(0.2),
        nn.BatchNorm1d(num_features=channels * 4),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d( channels*4,  channels*8, kernel_size= 3),
        nn.ReLU(),
        
        #  nn.Dropout1d(0.2),
        nn.BatchNorm1d(num_features=channels * 8),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(channels*8,  channels * 16, kernel_size= 3),
        nn.ReLU(),
        
        #  nn.Dropout1d(0.2),
        nn.BatchNorm1d(num_features=channels * 16),
        nn.MaxPool1d(kernel_size = 2),

        nn.Conv1d(channels * 16,  channels * 32, kernel_size= 3),
        nn.ReLU(),
        
        #  nn.Dropout1d(0.2),
        nn.BatchNorm1d(num_features=channels * 32),
        nn.MaxPool1d(kernel_size = 2),

        nn.Conv1d(channels * 32,  channels * 64, kernel_size= 3),
        nn.ReLU(),
        
        #  nn.Dropout1d(0.2),
        nn.BatchNorm1d(num_features=channels * 64),
        nn.MaxPool1d(kernel_size = 2),

        nn.Conv1d(channels * 64,  channels * 128, kernel_size= 3),
        nn.ReLU(),
        
        #  nn.Dropout1d(0.2),
        nn.BatchNorm1d(num_features=channels * 128),
        nn.MaxPool1d(kernel_size = 2),
        
        )

        self.classifier = nn.Sequential(
            nn.Linear(3072, 2000),
            # nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.ReLU(inplace=1),
            nn.Linear(1000, 80),
            nn.ReLU(inplace= 1),
            nn.Linear(80, 1),
        )


    def forward(self, x):
        x = self.seq(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Transformer1d(nn.Module):
    def __init__(self, input_size, n_classes, n_length, d_model, nhead, dim_feedforward, dropout, activation='relu'):
        super(Transformer1d, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.n_length = n_length
        self.d_model = d_model

        # Assuming input_size is not necessarily equal to d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Calculate the output size after the Transformer
        self.fc_out_size = d_model * n_length

        self.fc = nn.Linear(self.fc_out_size, n_classes)

    def forward(self, x):
       

        # Project input to d_model dimension
        x = x.permute(2, 0, 1)  # Change shape to (n_length, batch_size, input_size)
        x = self.input_projection(x)  # Shape becomes (n_length, batch_size, d_model)

        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)  # Shape remains (n_length, batch_size, d_model)

        # Flatten the output
        x = x.permute(1, 2, 0)  
        
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch_size, d_model * n_length)
        
        x = self.fc(x)  
        return x


class Bio(nn.Module):
    def __init__(self, input_size=32, feature_size=64):
        super(Bio, self).__init__()

        self.features = Transformer1d(
            input_size= 8, # for brainwaves data
            n_classes=64,
            n_length=750,
            d_model=32,
            nhead=8,
            dim_feedforward=128,
            dropout=0.3,
            activation='relu'
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=1),
            nn.Linear(64, 20),
            nn.ReLU(inplace= 1),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # x = self.cnn1d(x)
        x = self.features(x)
        x = self.classifier(x)

        return x
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))

class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        
        # Bottleneck convolution layers to reduce dimensionality
        self.bottleneck = nn.Conv1d(in_channels, 12, kernel_size=1, stride=1, padding=0)
        
        # Convolutional branches with different kernel sizes
        self.conv1 = nn.Conv1d(12, 12, kernel_size=25, stride=1, padding=12)
        self.conv2 = nn.Conv1d(12, 12, kernel_size=75, stride=1, padding=37)
        self.conv3 = nn.Conv1d(12, 12, kernel_size=125, stride=1, padding=62)
        
        # Pooling layer followed by a convolution
        self.pool = nn.MaxPool1d(kernel_size=25, stride=1, padding=12)
        self.pool_conv = nn.Conv1d(in_channels, 12, kernel_size=1, stride=1, padding=0)
        
        # Batch Normalization and Activation
        self.batch_norm = nn.BatchNorm1d(48)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # He initialization for ReLU activation layers
        nn.init.kaiming_uniform_(self.bottleneck.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.pool_conv.weight, mode='fan_in', nonlinearity='relu')
        
        # Initialize biases to zero
        nn.init.zeros_(self.bottleneck.bias)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.pool_conv.bias)

    def forward(self, x):
        # Bottleneck output
        bottleneck_output = self.bottleneck(x)
        # Different convolution branches
        conv1_output = self.conv1(bottleneck_output)

        conv2_output = self.conv2(bottleneck_output)

        conv3_output = self.conv3(bottleneck_output)
  
        # Pooling branch
        pool_output = self.pool(x)
        pool_conv_output = self.pool_conv(pool_output)
      
        # Concatenate outputs
        concat_output = torch.cat([conv1_output, conv2_output, conv3_output, pool_conv_output], dim=1)
        
        # Batch Normalization and Activation
        output = F.relu(self.batch_norm(concat_output))
        
        return output

class ResidualModule(nn.Module):
    def __init__(self, channels):
        super(ResidualModule, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm1d(channels)
      # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # He initialization for ReLU activation
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        # Residual connection
        residual = self.conv(x)
        residual = self.batch_norm(residual)
        output = F.relu(residual + x)
        return output

class EEGInceptionModel(nn.Module):
    def __init__(self, in_channels = 8):
        super(EEGInceptionModel, self).__init__()
        
        self.channels = in_channels
        # Initial Inception Module
        self.initial_inception = InceptionModule(in_channels=self.channels)
        

        # Intermediate Inception Modules
        self.inception_modules = nn.ModuleList([InceptionModule(in_channels=48) for _ in range(5)])
        
        # Residual Modules
        self.residual1 = ResidualModule(channels=48)
        self.residual2 = ResidualModule(channels=48)
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(48, 1), 
            nn.Sigmoid())
        
        
        # Initialize the final layer with Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Xavier initialization for the final layer (good for sigmoid activation)
        if hasattr(self.fc[0], 'weight'):
            nn.init.xavier_uniform_(self.fc[0].weight)
            nn.init.zeros_(self.fc[0].bias)
    
    def forward(self, x):
        # Pass through initial Inception module
        x = self.initial_inception(x)
        
        # Pass through alternating Inception and Residual modules
        for i, inception_module in enumerate(self.inception_modules):
            x = inception_module(x)
            if i == 1:
                x = self.residual1(x)
            elif i == 4:
                x = self.residual2(x)
            
        
        # Global average pooling
        x = self.avg_pool(x).squeeze(-1)
   
        # Linear layer for classification
        output = self.fc(x)
        
        return output
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Block(nn.Module):
    def __init__(self, inplace):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=inplace, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=inplace, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=inplace, out_channels=16, kernel_size=8, stride=2, padding=3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class ChronoNet(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # Use fewer blocks and channels
        self.block1 = Block(channel)
        self.block2 = Block(48)  # Adjust input to match output channels from Block
        
        # Use a single GRU layer to simplify the model
        self.gru = nn.GRU(input_size=48, hidden_size=32, batch_first=True)
        
        # Linear layer for output prediction
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.permute(0, 2, 1)  # Adjust for GRU input
        
        gru_out, _ = self.gru(x)
        # Use the last output from GRU
        x = gru_out[:, -1, :]  
        x = self.fc(x)
        return x