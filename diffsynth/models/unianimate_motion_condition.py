from torch import nn

def create_motion_condition_layers():
    concat_dim = 4
    dwpose_embedding = nn.Sequential(
                nn.Conv3d(3, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, 5120, (1,2,2), stride=(1,2,2), padding=0))

    randomref_dim = 20
    randomref_embedding_pose = nn.Sequential(
                nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
                )
    return dwpose_embedding, randomref_embedding_pose
