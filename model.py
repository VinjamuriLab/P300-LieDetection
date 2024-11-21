import torch.nn as nn
import torch
import time

from torchvision.models import vit_l_32

class Block(nn.Module):
    def __init__(self, inplace):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=inplace, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=inplace, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=inplace, out_channels=32, kernel_size=8, stride=2, padding=3)
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
        self.block1 = Block(channel)
        self.block2 = Block(96)
        self.block3 = Block(96)
        
        # Adjust GRU layers to handle the updated input dimensions
        self.gru1 = nn.GRU(input_size=96, hidden_size=32, batch_first=True)
        self.gru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
        self.gru3 = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        self.gru4 = nn.GRU(input_size=96, hidden_size=32, batch_first=True)
        
        self.gru_linear = nn.Linear(112, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        
        gru_out1, _ = self.gru1(x)
        gru_out2, _ = self.gru2(gru_out1)
        gru_out = torch.cat([gru_out1, gru_out2], dim=2)
        # print(gru_out.shape, "1")

        gru_out3, _ = self.gru3(gru_out)
        gru_out = torch.cat([gru_out1, gru_out2, gru_out3], dim=2)
        # print(gru_out.shape, "b")
        linear_out = self.relu(self.gru_linear(gru_out.permute(0, 2, 1)))
        gru_out4, _ = self.gru4(linear_out.permute(0, 2, 1))
        # print(gru_out.shape, "c")
        x = self.flatten(gru_out4)
        x = self.fc1(x)
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
    
class ViT1DEEGClassifier(nn.Module):
    def __init__(self, input_channels=8, input_length=1800, num_classes=2):
        super(ViT1DEEGClassifier, self).__init__()

        # Load ViT and modify for 1D EEG input
        self.vit = vit_l_32(pretrained=True)
        self.vit.conv_proj = nn.Conv1d(input_channels, 768, kernel_size=16, stride=16)
        self.vit.heads.head = nn.Linear(768, num_classes)
        
    def forward(self, x):
        x = self.vit(x)
        return x
    
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
        # This step is crucial and needs to be adjusted based on how you handle the sequence
        # For simplicity, assuming the output size remains d_model * n_length
        # This might need adjustment based on your actual sequence handling (e.g., pooling, averaging)
        self.fc_out_size = d_model * n_length
        # print("fc_out", self.fc_out_size)
        # Final linear layer to match the desired feature size (n_classes)
        self.fc = nn.Linear(self.fc_out_size, n_classes)

    def forward(self, x):
        # Input x shape: (batch_size, input_size, n_length)

        # Project input to d_model dimension
        x = x.permute(2, 0, 1)  # Change shape to (n_length, batch_size, input_size)
        x = self.input_projection(x)  # Shape becomes (n_length, batch_size, d_model)

        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)  # Shape remains (n_length, batch_size, d_model)

        # Flatten the output
        x = x.permute(1, 2, 0)  # Change shape to (batch_size, d_model, n_length)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch_size, d_model * n_length)
        # print(x.shape, "oyasid hpd asodih")
        # Final linear layer
        x = self.fc(x)  # Shape becomes (ba||tch_size, n_classes)

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
        x = x.squeeze(-1)
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



class Conv1D_v1(nn.Module):
    def __init__(self, channels = 8):
        super(Conv1D_v1, self).__init__() 

        self.seq = nn.Sequential(
         
         nn.Conv1d(channels, 16, kernel_size= 3),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size= 2), 

        nn.Conv1d(16, 32, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(32, 64, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(64, 128, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size = 2)
        )

        
        self.classifier = nn.Sequential(
            nn.Linear(6912, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.ReLU(inplace=1),
            nn.Linear(1000, 80),
            nn.ReLU(inplace= 1),
            nn.Linear(80, 1), 
            nn.Sigmoid()
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
import torch
import torch.nn as nn
import time

class Conv1D_LSTM(nn.Module):
    def __init__(self, channels=8, lstm_hidden_size=128, lstm_layers=2):
        super(Conv1D_LSTM, self).__init__()

        # Convolutional layers
        self.seq = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # LSTM layer
        # self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(6656, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass through convolutional layers
        x = self.seq(x)
        
        # Permute the output to (batch_size, sequence_length, feature_size) for LSTM
        # x = x.permute(0, 2, 1)  # (batch_size, feature_size, sequence_length) -> (batch_size, sequence_length, feature_size)
        
        # Pass through LSTM
        # lstm_out, (hn, cn) = self.lstm(x)

        # Only use the output of the last LSTM step
        # lstm_out = hn[-1]

        x = x.view(x.size(0), -1)
        # print(x.shape)
        # Pass through the classifier
        x = self.classifier(x)

        return x

    def save(self, name=None):
        """
        Save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))

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


        # out_features = 5376
        out_features = 7424
        self.classifier = nn.Sequential(
            nn.Linear(3072, 2000),
            # nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.ReLU(inplace=1),
            nn.Linear(1000, 80),
            nn.ReLU(inplace= 1),
            nn.Linear(80, 1),
            # nn.Sigmoid()
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

class Conv1D_v2v2(nn.Module):
    def __init__(self, channels = 8):
        super(Conv1D_v2v2, self).__init__() 

        self.seq = nn.Sequential(
         
         nn.Conv1d(channels, channels * 2, kernel_size= 3),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size= 2), 

        nn.Conv1d( channels* 2, channels * 4, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d( channels*4,  channels*8, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(channels*8,  channels * 16, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size = 2),

        # nn.Conv1d(channels * 16,  channels * 32, kernel_size= 3),
        # nn.ReLU(),
        # nn.MaxPool1d(kernel_size = 2),

        # nn.Conv1d(channels * 32,  channels *64, kernel_size= 3),
        # nn.ReLU(),
        # nn.MaxPool1d(kernel_size = 2),

        # nn.Conv1d(channels * 32,  channels *128, kernel_size= 3),
        # nn.ReLU(),
        # nn.MaxPool1d(kernel_size = 2)

        )


        # out_features = 6656
        out_features = 7680
        self.classifier = nn.Sequential(
            nn.Linear(out_features, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.ReLU(inplace=1),
            nn.Linear(1000, 80),
            nn.ReLU(inplace= 1),
            nn.Linear(80, 1),
            nn.Sigmoid()
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

class Conv1D_v3(nn.Module):
    def __init__(self, channels ):
        super(Conv1D_v3, self).__init__() 

        self.seq = nn.Sequential(
         
         nn.Conv1d(8, 16, kernel_size= 3),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size= 2), 

        nn.Conv1d(16, 32, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(32, 64, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 2),

        nn.Conv1d(64, 128, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size = 2),

        nn.Conv1d(128, 256, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size = 2)


        )

        self.classifier = nn.Sequential(
            nn.Linear(7424, 2000),
            # nn.Dropout(p = 0.3), 
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            # nn.Dropout(p = 0.3), 
            nn.ReLU(inplace=1),
            nn.Linear(1000, 80),
            # nn.Dropout(p = 0.3), 
            nn.ReLU(inplace= 1),
            nn.Linear(80, 1),
            nn.Sigmoid()
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






class Conv1D_v4(nn.Module):
    def __init__(self, channels, kernel_size = 3, padding = 1, stride = 2):
        super(Conv1D_v4, self).__init__() 
 
        self.seq = nn.Sequential(
         
            self.get_layer(channels, channels * 2, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size ),
            
            self.get_layer(channels * 2, channels * 4, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            
            self.get_layer(channels * 4, channels * 2, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            
            self.get_layer(channels * 2, channels, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            
            self.get_layer(channels , 1, stride = 1, padding = 1, final_layer= 1, kernel_size= 2),

            # self.get_layer(channels//2, 1, stride = stride, padding = padding, final_layer= True, kernel_size= kernel_size),

        )

    def get_layer(self, input_c, output_c, stride, padding, final_layer, kernel_size):

        if not final_layer:
            return nn.Sequential(
                nn.Conv1d(input_c, output_c, kernel_size=kernel_size,padding = padding, stride = stride),
                nn.BatchNorm1d(output_c),
                nn.MaxPool1d(stride = stride, kernel_size=kernel_size),
                nn.LeakyReLU(0.2)
                
            )

        return nn.Sequential(
            nn.Conv1d(input_c, output_c, kernel_size=kernel_size,padding = padding, stride = stride),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, input1):
        
      
        x = self.seq(input1)
       
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


class Conv1D_v4v2(nn.Module):
    def __init__(self, channels, kernel_size = 3, padding = 1, stride = 2):
        super(Conv1D_v4v2, self).__init__() 
 
        self.seq = nn.Sequential(
         
            self.get_layer(channels, channels * 2, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size ),
            
            self.get_layer(channels * 2, channels * 4, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            
            self.get_layer(channels * 4, channels * 2, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            
            self.get_layer(channels * 2, channels, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            

            # self.get_layer(channels//2, 1, stride = stride, padding = padding, final_layer= True, kernel_size= kernel_size),

        )

        self.seq2 = nn.Sequential(
            self.get_layer(channels , 1, stride = 1, padding = 1, final_layer= 1, kernel_size= 2),
        )

    def get_layer(self, input_c, output_c, stride, padding, final_layer, kernel_size):

        if not final_layer:
            return nn.Sequential(
                nn.Conv1d(input_c, output_c, kernel_size=kernel_size,padding = padding, stride = stride),
                nn.BatchNorm1d(output_c),
                nn.MaxPool1d(stride = stride, kernel_size=kernel_size),
                nn.LeakyReLU(0.2)
                
            )
        return nn.Sequential(

            nn.Linear(24, 8),
            nn.ReLU(), 
            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self, input1):
        
      
        x = self.seq(input1)

       
        x = x.reshape(-1, 24)
        x = self.seq2(x)
        
       
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

import torch.nn as nn 
import torch

class Conv1D_v5(nn.Module):
    def __init__(self, channels = 8, kernel_size = 3, padding = 1, stride = 2):
        super(Conv1D_v5, self).__init__() 

        self.seq = nn.Sequential(
         
            self.get_layer(channels, channels * 2, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size ),
            
            self.get_layer(channels * 2, channels * 4, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),
            
            self.get_layer(channels * 4, channels * 2, stride = stride, padding = padding, final_layer= False, kernel_size= kernel_size),

        )

        self.linear = nn.Sequential(
            nn.Linear(177, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(), 
            nn.Linear(20,1),
            nn.Sigmoid()
        )

    def get_layer(self, input_c, output_c, stride, padding, final_layer, kernel_size):

        if not final_layer:
            return nn.Sequential(
                nn.Conv1d(input_c, output_c, kernel_size=kernel_size,padding = padding, stride = stride),
                nn.BatchNorm1d(output_c),
                nn.MaxPool1d(stride = stride, kernel_size=kernel_size),
                nn.LeakyReLU(0.2)
                
            )

    def forward(self, z):
        
        input1 = z[0]
        input2 = z[1]
       
        x = self.seq(input1)
          # Concatenate with the scalar input
        # print("asdiufhasi udhfpioasudhf poasiudfoiasu odsih ")
        # print(x.shape, x.shape[0])
        x =torch.flatten(x, start_dim=1, end_dim=-1 )
      
        input2 = input2.unsqueeze(dim = 1)
        x = torch.cat((x, input2), dim=1)  
        # Concatenate along the feature dimension
        # print(x.shape, "here")
        x = self.linear(x)
        # print(x.shape)
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

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, num_channels, num_classes=2):
        super(ResNet1D, self).__init__()
        self.in_channels = num_channels
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
       
        out = out.reshape(out.shape[0])
        out = torch.sigmoid(out)
        return out


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


