import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import FFN
from .sparsebev_transformer import SparseBEVSelfAttention, SparseBEVSampling, AdaptiveMixing
from .utils import DUMP, generate_grid, batch_indexing
from .bbox.utils import encode_bbox
import torch.nn.functional as F


def index2point(coords, pc_range, voxel_size):
    """
    coords: [B, N, 3], int
    pc_range: [-40, -40, -1.0, 40, 40, 5.4]
    voxel_size: float
    """
    coords = coords * voxel_size
    coords = coords + torch.tensor(pc_range[:3], device=coords.device)
    return coords


def point2bbox(coords, box_size):
    """
    coords: [B, N, 3], float
    box_size: float
    """
    wlh = torch.ones_like(coords.float()) * box_size
    bboxes = torch.cat([coords, wlh], dim=-1)  # [B, N, 6]
    return bboxes


def upsample(pre_feat, pre_coords, interval):
    '''
    :param pre_feat: (Tensor), features from last level, (B, N, C)
    :param pre_coords: (Tensor), coordinates from last level, (B, N, 3) (3: x, y, z)
    :param interval: interval of voxels, interval = scale ** 2
    :param num: 1 -> 8
    :return: up_feat : upsampled features, (B, N*8, C//8)
    :return: up_coords: upsampled coordinates, (B, N*8, 3)
    '''
    pos_list = [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    bs, num_query, num_channels = pre_feat.shape
    
    up_feat = pre_feat.reshape(bs, num_query, 8, num_channels // 8)  # [B, N, 8, C/8]
    up_coords = pre_coords.unsqueeze(2).repeat(1, 1, 8, 1).contiguous()  # [B, N, 8, 3]
    for i in range(len(pos_list)):
        up_coords[:, :, i + 1, pos_list[i]] += interval

    up_feat = up_feat.reshape(bs, -1, num_channels // 8)
    up_coords = up_coords.reshape(bs, -1, 3)

    return up_feat, up_coords

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class SparseVoxel3DEncoder(nn.Module):
    def __init__(self, embed_dims=[192, 64], vel_dims=2):
        super().__init__()
        self.pos_fc = nn.Sequential(*linear_relu_ln(embed_dims[0], 1, 2, input_dims=4))
        self.size_fc = nn.Sequential(*linear_relu_ln(embed_dims[1], 1, 2, input_dims=3))

    @torch.no_grad()
    def init_weights(self):
        for fc in [self.pos_fc, self.size_fc]:
            for m in fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, box_3d, time_state):
        pos_feat = self.pos_fc(torch.cat([box_3d[..., 0:3], time_state], dim=-1))
        size_feat = self.size_fc(box_3d[..., 3:6])
        return torch.cat([pos_feat, size_feat], dim=-1)
    

class SparseVoxelDecoder(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 num_layers=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 semantic=False,
                 topk_training=None,
                 topk_testing=None,
                 pc_range=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.semantic = semantic
        self.voxel_dim = [200, 200, 16]
        self.topk_training = topk_training
        self.topk_testing = topk_testing

        self.decoder_layers = nn.ModuleList()
        self.lift_feat_heads = nn.ModuleList()
        #self.occ_pred_heads = nn.ModuleList()
        
        if semantic:
            self.seg_pred_heads = nn.ModuleList()

        for i in range(num_layers):
            self.decoder_layers.append(SparseVoxelDecoderLayer(
                 embed_dims=embed_dims,
                 num_frames=num_frames,
                 num_points=num_points // (2 ** i),
                 num_groups=num_groups,
                 num_levels=num_levels,
                 pc_range=pc_range,
                 self_attn=i in [0, 1]
            ))
            self.lift_feat_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 8),
                nn.ReLU(inplace=True)
            ))
            #self.occ_pred_heads.append(nn.Linear(embed_dims, 1))

            if semantic:
                self.seg_pred_heads.append(nn.Linear(embed_dims, num_classes))

    @torch.no_grad()
    def init_weights(self):
        for i in range(len(self.decoder_layers)):
            self.decoder_layers[i].init_weights()

    def forward(self, mlvl_feats, img_metas, prev_pos=None, prev_feat=None):
        occ_preds = []
        
        topk = self.topk_training if self.training else self.topk_testing
        
        B = len(img_metas)
        # init query coords
        interval = 2 ** self.num_layers
        query_coord = generate_grid(self.voxel_dim, interval).expand(B, -1, -1)  # [B, N, 3]
        query_feat = torch.zeros([B, query_coord.shape[1], self.embed_dims], device=query_coord.device)  # [B, N, C]
        curr_time_state = torch.zeros([B, query_coord.shape[1], 1], device=query_coord.device)

        if prev_pos is not None and prev_feat is not None:
            # Scale the fine-grained historical coordinates to the current coarse interval
            prev_occ_loc_coarse = torch.div(prev_pos, interval, rounding_mode='trunc').long()
            prev_time_state = torch.ones([B, prev_occ_loc_coarse.shape[1], 1], device=query_coord.device)

            # Concatenate historical queries with current uniform grid queries
            query_coord = torch.cat([query_coord, prev_occ_loc_coarse], dim=1)
            query_feat = torch.cat([query_feat, prev_feat], dim=1)
            time_state = torch.cat([curr_time_state, prev_time_state], dim=1)

            num_curr = curr_time_state.shape[1]
            num_total = time_state.shape[1]
            attn_mask = torch.zeros((num_total, num_total), dtype=torch.bool, device=query_coord.device)
            attn_mask[num_curr:, :num_curr] = True  # Prev (row) cannot see Curr (col)
            
            attn_mask_inf = torch.zeros_like(attn_mask, dtype=torch.float)
            attn_mask_inf[attn_mask] = float('-inf')
            attn_mask = attn_mask_inf
        else:
            time_state = curr_time_state
            attn_mask = None

        for i, layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i
            
            interval = 2 ** (self.num_layers - i)  # 8 4 2 1
            
            # bbox from coords
            query_bbox = index2point(query_coord, self.pc_range, voxel_size=0.4)  # [B, N, 3]
            query_bbox = point2bbox(query_bbox, box_size=0.4 * interval)  # [B, N, 6]
            query_bbox = encode_bbox(query_bbox, pc_range=self.pc_range)  # [B, N, 6]

            # transformer layer
            query_feat = layer(query_feat, query_bbox, mlvl_feats, img_metas, time_state, attn_mask)  # [B, N, C]
            
            if i == 0: 
                attn_mask = None
        
            # upsample 2x
            query_feat = self.lift_feat_heads[i](query_feat)  # [B, N, 8C]
            query_feat_2x, query_coord_2x = upsample(query_feat, query_coord, interval // 2)

            if self.semantic:
                seg_pred_2x = self.seg_pred_heads[i](query_feat_2x)  # [B, K, CLS]
            else:
                seg_pred_2x = None

            # sparsify after seg_pred
            non_free_prob = 1 - F.softmax(seg_pred_2x, dim=-1)[..., -1]  # [B, K]
            indices = torch.topk(non_free_prob, k=topk[i], dim=1)[1]  # [B, K]

            query_coord_2x = batch_indexing(query_coord_2x, indices, layout='channel_last')  # [B, K, 3]
            query_feat_2x = batch_indexing(query_feat_2x, indices, layout='channel_last')  # [B, K, C]
            seg_pred_2x = batch_indexing(seg_pred_2x, indices, layout='channel_last')  # [B, K, CLS]

            time_state = torch.zeros([B, query_coord_2x.shape[1], 1], device=query_coord.device)

            occ_preds.append((
                torch.div(query_coord_2x, interval // 2, rounding_mode='trunc').long(),
                None,
                seg_pred_2x,
                query_feat_2x,
                interval // 2)
            )

            query_coord = query_coord_2x.detach()
            query_feat = query_feat_2x.detach()

        return occ_preds


class SparseVoxelDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 pc_range=None,
                 self_attn=True):
        super().__init__()

        self.position_encoder = SparseVoxel3DEncoder(embed_dims=[192, 64])

        if self_attn:
            self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range, scale_adaptive=True)
            self.norm1 = nn.LayerNorm(embed_dims)
        else:
            self.self_attn = None
        
        self.sampling = SparseBEVSampling(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range
        )
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=num_groups,
            out_points=num_points * num_frames * num_groups
        )
        self.ffn = FFN(embed_dims, feedforward_channels=embed_dims * 2, ffn_drop=0.1)
        
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    @torch.no_grad()
    def init_weights(self):
        self.position_encoder.init_weights()
        if self.self_attn is not None:
            self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        self.ffn.init_weights()

    def forward(self, query_feat, query_bbox, mlvl_feats, img_metas, time_state, attn_mask=None):
        query_pos = self.position_encoder(query_bbox, time_state)
        query_feat = query_feat + query_pos

        if self.self_attn is not None:
            query_feat = self.norm1(self.self_attn(query_bbox, query_feat, pre_attn_mask=attn_mask))
        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        return query_feat
