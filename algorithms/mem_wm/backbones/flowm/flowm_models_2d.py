import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
import lightning as L

def roll_batch(h, shift):
    """
    h:     [B, C, H, W]
    shift: [B, 2]   (dx, dy)  ← same order as action
    """
    B, _, _, _ = h.shape
    rolled = []
    for b in range(B):
        dx, dy = shift[b]                 # two scalars
        # slice gives [C, H, W] so dims become (1, 2)
        rolled.append(torch.roll(h[b],
                                 shifts=(int(dy.item()), int(dx.item())),
                                 dims=(1, 2)))
    return torch.stack(rolled, dim=0)     # [B, C, H, W]

def roll_batch_with_v(h, shift):
    """
    h:     [B, V, C, H, W]
    shift: [B, 2]   (dx, dy)  ← same order as action
    """
    B, V, _, _, _ = h.shape
    rolled = []
    for b in range(B):
        dx, dy = shift[b]                 # two scalars
        # slice gives [V, C, H, W] so dims become (1, 2)
        rolled.append(torch.roll(h[b],
                                 shifts=(int(dy.item()), int(dx.item())),
                                 dims=(2, 3)))
    return torch.stack(rolled, dim=0)     # [B, V, C, H, W]


# FERNN Cell with only self-motion equivariance (no velocity channels)
# Currently not used, just included for reference
class FERNN_Cell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3, world_size=50, window_size=28, use_mlp_encoder=False, self_motion_equivariance=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.world_size = world_size
        self.window_size = window_size
        self.self_motion_equivariance = self_motion_equivariance

        # circular convs without bias
        u_pad = u_kernel_size // 2
        h_pad = h_kernel_size // 2
        self.use_mlp_encoder = use_mlp_encoder
        
        if use_mlp_encoder:
            # Actually just a linear encoder (can be replaced with MLP)
            input_size = input_channels * self.window_size * self.window_size
            hidden_size = hidden_channels * self.window_size * self.window_size
            self.linear_u = nn.Sequential(
                nn.Linear(input_size, hidden_size),
            )
        else:
            self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                 padding=u_pad, padding_mode='circular', bias=False)

        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                 padding=h_pad, padding_mode='circular', bias=False)
        self.activation = nn.ReLU()



    def forward(self, u_t, h_prev, action):
        # u_t: (batch, C, H, W)
        # h_prev: (batch, hidden, H, W)
        
        # conv_u then expand
        B, _, _, _ = u_t.size()
        if self.use_mlp_encoder:
            u_linear = self.linear_u(u_t.view(B, -1))
            u_conv = u_linear.view(B, self.hidden_channels, self.window_size, self.window_size)
        else:
            u_conv = self.conv_u(u_t)

        B, C, H, W = u_conv.size()
        B, C, H_hidden, W_hidden = h_prev.size()

        # Embed the processed input in the world size
        u_full = torch.zeros(B, C, self.world_size, self.world_size).to(h_prev.device)
        u_full[:, :, :H, :W] = u_conv # Place the processed input in the top left corner of the world (the current window)

        if self.self_motion_equivariance:
            h_shifted = roll_batch(h_prev, -1 * action)
        else:
            h_shifted = h_prev

        h_conv = self.conv_h(h_shifted)

        # combine and activate
        h_next = self.activation(u_full + h_conv)
        return h_next


# FERNN Cell with velocity channels and self-motion equivariance (main cell used in the paper)
class FERNN_Cell_w_VChannels(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3, world_size=50, window_size=28, use_mlp_encoder=False,
                 v_range=0, self_motion_equivariance=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.world_size = world_size
        self.window_size = window_size
        self.v_range = v_range
        self.v_list = [(vx, vy) for vx in range(-v_range, v_range + 1) for vy in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)
        self.self_motion_equivariance = self_motion_equivariance

        # circular convs without bias
        u_pad = u_kernel_size // 2
        h_pad = h_kernel_size // 2
        self.use_mlp_encoder = use_mlp_encoder
        
        if use_mlp_encoder:
            # Actually just a linear encoder (can be replaced with MLP)
            input_size = input_channels * self.window_size * self.window_size
            hidden_size = hidden_channels * self.window_size * self.window_size
            self.linear_u = nn.Sequential(
                nn.Linear(input_size, hidden_size),
            )
        else:
            self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                 padding=u_pad, padding_mode='circular', bias=False)

        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                 padding=h_pad, padding_mode='circular', bias=False)
        self.activation = nn.ReLU()

    def forward(self, u_t, h_prev, action):
        # u_t: (batch, C, H, W)
        # h_prev: (batch, velocity_channels, hidden_channels, H, W)
        
        # conv_u then expand
        B, _, _, _ = u_t.size()
        if self.use_mlp_encoder:
            u_linear = self.linear_u(u_t.view(B, -1))
            u_conv = u_linear.view(B, self.hidden_channels, self.window_size, self.window_size)
        else:
            u_conv = self.conv_u(u_t)

        B, C, H, W = u_conv.size()
        B, V, C, H_hidden, W_hidden = h_prev.size()

        # Embed the processed input in the world size
        u_full = torch.zeros(B, C, self.world_size, self.world_size).to(h_prev.device)
        u_full[:, :, :H, :W] = u_conv # Place the processed input in the top left corner of the world (the current window)
        u_full = u_full.unsqueeze(1).expand(-1, self.num_v, -1, -1, -1)

        # shift hidden via torch.roll per velocity channel (intrinsic equivariance)
        h_shift = []
        for i, (vx, vy) in enumerate(self.v_list):
            h_shift.append(torch.roll(h_prev[:, i], shifts=(vy, vx), dims=(2, 3)))
        h_shift = torch.stack(h_shift, dim=1)  # (batch, num_v, hidden, H, W)

        # shift based on ego motion
        if self.self_motion_equivariance:
            h_shifted = roll_batch_with_v(h_shift, -1 * action)
        else:
            h_shifted = h_shift
        
        h_shifted = h_shifted.view(B * V, C, H_hidden, W_hidden)
        h_conv = self.conv_h(h_shifted)
        h_conv = h_conv.view(B, V, C, H_hidden, W_hidden)

        # combine and activate
        h_next = self.activation(u_full + h_conv)
        return h_next


# FERNN Cell with no velocity channels, no self-motion equivariance, instead concatenating action to hidden state and input
class FERNN_Cell_ActionConcat(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3, world_size=50, window_size=28, use_mlp_encoder=False,
                 concat_action_to_hidden=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.world_size = world_size
        self.window_size = window_size
        self.use_mlp_encoder = use_mlp_encoder
        self.concat_action_to_hidden = concat_action_to_hidden
        self.num_v = None
        
        # circular convs without bias
        u_pad = u_kernel_size // 2
        h_pad = h_kernel_size // 2
        
        # Add 2 channels for action (dx, dy)
        action_channels = 2
        input_with_action_channels = input_channels + action_channels

        if use_mlp_encoder:
            input_size = input_with_action_channels * self.window_size * self.window_size
            hidden_size = hidden_channels * self.window_size * self.window_size
            self.linear_u = nn.Sequential(
                nn.Linear(input_size, hidden_size),
            )
        else:
            self.conv_u = nn.Conv2d(input_with_action_channels, hidden_channels, u_kernel_size,
                                 padding=u_pad, padding_mode='circular', bias=False)

        # If concatenating action to hidden, increase input channels for conv_h
        if self.concat_action_to_hidden:
            hidden_in_channels = hidden_channels + action_channels
        else:
            hidden_in_channels = hidden_channels

        self.conv_h = nn.Conv2d(hidden_in_channels, hidden_channels, h_kernel_size,
                                 padding=h_pad, padding_mode='circular', bias=False)
        self.activation = nn.ReLU()

    def forward(self, u_t, h_prev, action):
        # u_t: (batch, C, H, W)
        # h_prev: (batch, hidden, H, W)
        # action: (batch, 2) - (dx, dy)
        
        B, C, H, W = u_t.size()
        
        # Create action channels by repeating action values across spatial dimensions
        # action: [B, 2] -> [B, 2, H, W]
        action_expanded = action.view(B, 2, 1, 1).expand(B, 2, H, W)
        
        # Concatenate input with action channels
        u_with_action = torch.cat([u_t, action_expanded], dim=1)  # [B, C+2, H, W]
        
        if self.use_mlp_encoder:
            u_linear = self.linear_u(u_with_action.view(B, -1))
            u_conv = u_linear.view(B, self.hidden_channels, self.window_size, self.window_size)
        else:
            u_conv = self.conv_u(u_with_action)

        B, C, H, W = u_conv.size()
        B, C, H_hidden, W_hidden = h_prev.size()

        # Embed the processed input in the world size
        u_full = torch.zeros(B, C, self.world_size, self.world_size).to(h_prev.device)
        u_full[:, :, :H, :W] = u_conv # Place the processed input in the top left corner of the world (the current window)
 
        # Optionally concatenate action to hidden state
        if self.concat_action_to_hidden:
            # Expand action to match hidden state spatial dims
            action_expanded_h = action.view(B, 2, 1, 1).expand(B, 2, H_hidden, W_hidden)
            h_input = torch.cat([h_prev, action_expanded_h], dim=1)  # [B, hidden+2, H, W]
        else:
            h_input = h_prev

        h_conv = self.conv_h(h_input)

        # combine and activate
        h_next = self.activation(u_full + h_conv)
        return h_next


class FlowEquivariantRNN(L.LightningModule):
    def __init__(self, input_channels, hidden_channels, world_size, window_size,
                 output_channels=None, h_kernel_size=3, u_kernel_size=3,
                 decoder_conv_layers=1, use_mlp_decoder=False, use_mlp_encoder=False,
                 cell_type='v_channels', v_range=0, self_motion_equivariance=True):
        super().__init__()
        self.world_size = world_size
        self.window_size = window_size
        self.output_channels = output_channels or input_channels
        self.use_mlp_decoder = use_mlp_decoder
        self.cell_type = cell_type
        self.v_range = v_range        
        self.self_motion_equivariance = self_motion_equivariance

        # Choose cell type based on flag
        if cell_type == 'v_channels':
            self.cell = FERNN_Cell_w_VChannels(
                input_channels, hidden_channels,
                h_kernel_size, u_kernel_size, world_size, window_size, use_mlp_encoder, v_range, self_motion_equivariance)
        elif cell_type == 'action_concat':
            self.cell = FERNN_Cell_ActionConcat(
                input_channels, hidden_channels,
                h_kernel_size, u_kernel_size, world_size, window_size, use_mlp_encoder)
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}. Must be 'v_channels' or 'action_concat'")
        
        self.num_v = self.cell.num_v

        self.hidden_channels = hidden_channels
        self.decoder_channels = hidden_channels # if max-pooling over v_channels
        # self.decoder_channels = hidden_channels * self.num_v if cell_type == 'v_channels' else hidden_channels

        if use_mlp_decoder:
            # MLP decoder: flatten hidden window and use linear layers
            input_size = hidden_channels * self.window_size * self.window_size
            output_size = self.output_channels * self.window_size * self.window_size
            self.decoder_mlp = nn.Linear(input_size, output_size)
        else:
            # Convolutional decoder
            decoder = []
            for _ in range(decoder_conv_layers):
                decoder += [nn.Conv2d(self.decoder_channels, self.decoder_channels, 3, padding=1, padding_mode='circular', bias=False), nn.ReLU()]
            decoder += [nn.Conv2d(self.decoder_channels, self.output_channels, 3, padding=1, padding_mode='circular', bias=False)]
            self.decoder_conv = nn.Sequential(*decoder)

    def forward(self, input_seq, input_actions, target_actions, target_seq=None, teacher_forcing_ratio=0.0, return_hidden=False, **kwargs):
        batch, T_in, C, H, W = input_seq.size()
        if H != self.window_size or W != self.window_size:
            raise ValueError(f"Input sequence height ({H}) and width ({W}) must match window size ({self.window_size})")
        
        device = input_seq.device
        pred_len = target_actions.size(1)

        if return_hidden:
            if self.cell_type == 'v_channels':
                input_seq_hiddens = torch.zeros(batch, T_in, self.num_v, self.hidden_channels, self.world_size, self.world_size, device=device)
                out_seq_hiddens = torch.zeros(batch, pred_len, self.num_v, self.hidden_channels, self.world_size, self.world_size, device=device)
            else:
                input_seq_hiddens = torch.zeros(batch, T_in, self.hidden_channels, self.world_size, self.world_size, device=device)
                out_seq_hiddens = torch.zeros(batch, pred_len, self.hidden_channels, self.world_size, self.world_size, device=device)

        # Initialize hidden state
        if self.cell_type == 'v_channels':
            h = torch.zeros(batch, self.num_v, self.hidden_channels, self.world_size, self.world_size, device=device)
        else:
            h = torch.zeros(batch, self.hidden_channels, self.world_size, self.world_size, device=device)

        # Encoder pass through cell
        for t in range(T_in):
            u_t = input_seq[:, t]
            h = self.cell(u_t, h, input_actions[:, t])

            if return_hidden:
                input_seq_hiddens[:, t] += h.detach()

        prev = input_seq[:, -1]
        outputs = []

        # Decoder
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                frame = target_seq[:, t]
            else:
                # frame = prev.detach() # optionally detach prev hidden state to reduce long gradient paths
                frame = prev
            h = self.cell(frame, h, target_actions[:, t])

            if return_hidden:
                out_seq_hiddens[:, t] += h.detach()

            # Extract the window from the hidden state
            if self.cell_type == 'v_channels':
                hidden_window = h[:, :, :, :self.window_size, :self.window_size]
            else:
                hidden_window = h[:, :, :self.window_size, :self.window_size]

            if self.use_mlp_decoder:
                # Flatten and use MLP decoder
                hidden_flat = hidden_window.reshape(batch, -1)
                out_flat = self.decoder_mlp(hidden_flat)
                out = out_flat.view(batch, self.output_channels, self.window_size, self.window_size)
            else:
                # Use convolutional decoder
                if self.cell_type == 'v_channels':
                    # Max pool over velocity channels pixel-wise
                    hidden_window = hidden_window.max(dim=1)[0].view(batch, -1, self.window_size, self.window_size)
                out = self.decoder_conv(hidden_window)
            
            outputs.append(out)
            prev = out

        if return_hidden:
            return torch.stack(outputs, dim=1), input_seq_hiddens, out_seq_hiddens # _, (B, T_in, num_v, C, H, W), (B, T_out, num_v, C, H, W)
        else:
            return torch.stack(outputs, dim=1)


class LinearRNNBaseline(L.LightningModule):
    def __init__(self, input_channels, hidden_dim, window_size, output_channels=None, use_mlp_decoder=False, concat_action_to_hidden=True):
        super().__init__()
        self.window_size = window_size
        self.output_channels = output_channels or input_channels
        self.use_mlp_decoder = use_mlp_decoder
        self.concat_action_to_hidden = concat_action_to_hidden

        if concat_action_to_hidden:
            self.action_channels = 2
            self.action_embedding = nn.Linear(self.action_channels, hidden_dim)
        
        # CNN encoder: 2 conv layers + flatten + linear
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.encoder_out_dim = 32 * window_size * window_size
        self.encoder_linear = nn.Linear(self.encoder_out_dim, hidden_dim)

        # Linear RNN
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)

        # Decoder (reuse FlowEquivariantRNN logic)
        if use_mlp_decoder:
            self.decoder = nn.Linear(hidden_dim, self.output_channels * window_size * window_size)
        else:
            self.decoder_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, self.output_channels, 3, padding=1)
            )

    def forward(self, input_seq, input_actions=None, target_actions=None, target_seq=None, teacher_forcing_ratio=0.0, return_hidden=False, **kwargs):
        # input_seq: [B, T_in, 1, H, W]
        B, T_in, C, H, W = input_seq.size()
        
        x = input_seq.view(B * T_in, C, H, W)
        x = self.encoder(x)
        x = x.view(B * T_in, -1)
        x = self.encoder_linear(x)
        x = x.view(B, T_in, -1)  # [B, T_in, hidden_dim]

        if self.concat_action_to_hidden:
            # Need to convert input_actions to dtype of action_embedding (float32 instead of int64)
            input_actions = self.action_embedding(input_actions.to(self.action_embedding.weight.dtype))
            x = x + input_actions # torch.cat([x, input_actions.unsqueeze(1)], dim=2)
    
        # RNN
        rnn_out, _ = self.rnn(x)  # [B, T_in, hidden_dim]

        # For prediction, use only the last hidden state as the initial state for decoding
        h = rnn_out[:, -1:, :]  # [B, 1, hidden_dim]
        pred_len = target_actions.size(1) if target_actions is not None else T_in
        outputs = []
        prev = h.squeeze(1)  # [B, hidden_dim]
        for t in range(pred_len):
            # For a simple baseline, just repeat the last hidden state
            out = prev
            if self.use_mlp_decoder:
                out_flat = self.decoder(out)
                out_img = out_flat.view(B, self.output_channels, self.window_size, self.window_size)
            else:
                # Expand dims for conv decoder
                out_img = self.decoder_conv(out.view(B, -1, 1, 1).expand(-1, -1, self.window_size, self.window_size))
            outputs.append(out_img)
            prev = out  # No recurrence in decoding for baseline
        return torch.stack(outputs, dim=1)  # [B, T_out, C, H, W]

