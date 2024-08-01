from globals import *

len_seq = 20_000

# Takes in full sequence output and predicts whether num generations is higher or lower
class KNet4(nn.Module):
    def __init__(self):
        super().__init__()


        hidden1 = 1000
        hidden2 = 100

        hidden3 = 50

        self.linear1 = nn.Linear((num_classes + 1)* len_seq, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, positions):

        # x  (batch, len_seq, num_classes)
        # positions (batch, len_seq)

        x = torch.cat((x, positions.unsqueeze(1)), dim=1)
        x = x.reshape(-1, (num_classes + 1) * len_seq)

        x = self.linear1(x)
        
        x = self.relu(x)

        x = self.linear2(x)

        x = self.relu(x)

        x = self.linear3(x)  # batch, num_classes, 1

        x = self.sigmoid(x)

        return x

