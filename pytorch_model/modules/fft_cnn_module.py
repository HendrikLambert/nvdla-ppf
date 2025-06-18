import torch
import torch.nn as nn
import math
import numpy as np
from modules.dft_helper import create_dft_matrix

import torch
import torch.nn as nn
import numpy as np


class ShuffleComplexPairsModule(nn.Module):
    """
    Shuffles pairs of channels (representing complex numbers) of an input
    tensor. It uses torch.split to get blocks of 2 channels, reorders them,
    and then uses torch.cat. This is much more efficient than splitting
    every individual channel.
    """

    def __init__(self, pair_permutation_indices: np.ndarray):
        """
        Args:
            pair_permutation_indices (np.ndarray): A 1D array of ints of size N.
                `pair_permutation_indices[k]` specifies the source *pair* index
                in the input tensor that should become the k-th *pair* in the
                output tensor.
        """
        super().__init__()

        self.indices = pair_permutation_indices
        self.num_pairs = pair_permutation_indices.shape[0]
        self.num_channels = self.num_pairs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input tensor has {x.shape[1]} channels, but this shuffler "
                f"was initialized for {self.num_channels} channels ({self.num_pairs} pairs)."
            )

        if self.num_pairs == 0:
            return x

        # Split the input tensor into pairs of channels (complex numbers)
        # Each element will have shape (B, 2, H, W).
        input_pair_slices = list(torch.split(x, 2, dim=1))

        # Prepare the list of output slices in the new order
        output_slices_in_new_order = [
            input_pair_slices[source_idx] for source_idx in self.indices
        ]

        # Concatenate the reordered slices along the channel dimension.
        return torch.cat(output_slices_in_new_order, dim=1)


class FFTCNNModule(nn.Module):
    """
    FFTCNNModule implements a Fast Fourier Transform (FFT) using a convolutional
    neural network (CNN) architecture. It uses grouped 1x1 convolutions to
    perform butterfly operations, and it applies a series of permutations
    to rearrange the input and output channels according to the Cooley-Tukey
    algorithm.

    The module is designed to compute the FFT of a complex input tensor.
    It expexts the numbers to be interleaved like [Re(X_0), Im(X_0), Re(X_1), Im(X_1), ...]
    It also expects the input tensor to have a shape of (B, 2*N, 1, 1), with N the amount of complex points.
    """

    def __init__(self, N: int):
        """
        Initializes the FFTCNNModule.

        - Initial bit-reversal permutation of the input channels.
        - A series of log2(N) stages. Each stage consists of:
            - An inter-stage permutation (shuffling).
            - A grouped 1x1 convolution to perform the butterfly operations.
        - A final unscrambling permutation, required because the convolution-based butterflies output pairs (X_k, X_{k+N/2}) together.

        Args:
            N (int): Number of complex points for the FFT. Must be a positive power of 2.
        """
        super().__init__()

        is_power_of_two = (N >= 2) and (math.log2(N).is_integer())
        if not is_power_of_two:
            raise ValueError("N must be a positive power of 2.")

        self.N = N
        self.num_features = 2 * N
        self.stages = int(math.log2(N))  # Round, because math.log returns float
        self.layers = nn.ModuleList()

        # First bit reversal, accounting for complex numbers
        bit_rev_perm_indices = self._create_bit_reversal_indices()
        self.layers.append(ShuffleComplexPairsModule(bit_rev_perm_indices))

        # Number of butterfly groups for the convolution layers.
        num_butterfly_groups_for_conv = self.N // 2

        # Cooley-Tukey Stages
        for stage_idx in range(self.stages):
            # This shuffle groups the outputs from the previous stage's DFTs
            # to form the inputs for the current stage's larger DFTs.
            if stage_idx > 0:
                perm_indices = self._create_inter_stage_indices(stage_idx)
                self.layers.append(ShuffleComplexPairsModule(perm_indices))

            # Grouped butterfly convolution for the current stage.
            W_gbf = self._create_butterfly_weights(stage_idx)
            conv_bf = nn.Conv2d(
                self.num_features,
                self.num_features,
                kernel_size=1,
                groups=num_butterfly_groups_for_conv,
                bias=False,
            )
            conv_bf.weight.data = W_gbf
            conv_bf.weight.requires_grad = False
            self.layers.append(conv_bf)

        # Final Unscrambling Permutation
        # Reorders channels from the final conv output [X_0, X_{N/2}, X_1, X_{N/2+1}, ...]
        # to the natural order [X_0, X_1, X_2, ...].
        final_perm_indices = self._create_final_unscramble_indices()
        self.layers.append(ShuffleComplexPairsModule(final_perm_indices))

    def _create_bit_reversal_indices(self) -> np.ndarray:
        """
        Creates permutation indices for the initial bit-reversal of complex pairs.
        Returns a permutation map of size N.
        """
        perm_indices = np.empty(self.N, dtype=int)
        num_bits = int(np.log2(self.N))

        # Iterate through each target complex pair index
        for target_pair_idx in range(self.N):
            val = target_pair_idx
            source_pair_idx = 0
            for _ in range(num_bits):
                source_pair_idx = (source_pair_idx << 1) | (val & 1)
                val >>= 1

            # The k-th target pair comes from the bit-reversed source pair
            perm_indices[target_pair_idx] = source_pair_idx

        return perm_indices

    def _create_inter_stage_indices(self, stage_idx: int) -> np.ndarray:
        """
        Creates permutation indices for complex pairs to wire inputs for the
        current stage's butterfly convolution.
        Returns a permutation map of size N.
        """
        perm_indices_pairs = np.empty(self.N, dtype=int)

        # Size of the DFTs computed in the PREVIOUS stage.
        prev_dft_size = 2**stage_idx

        # We iterate through each of the N/2 butterfly operations in the current stage.
        # k_bf_target is the physical index of the butterfly group (which takes 2 pairs).
        for k_bf_target in range(self.N // 2):
            # Map the physical butterfly index to its logical FFT context.
            dft_group_idx = k_bf_target // prev_dft_size
            k_within_dft = k_bf_target % prev_dft_size

            # Identify the two source DFTs from the previous stage.
            src_dft_group_E_idx = 2 * dft_group_idx
            src_dft_group_O_idx = 2 * dft_group_idx + 1

            # Determine if the source is an XE or XO output from a source butterfly.
            is_second_half = k_within_dft >= (prev_dft_size // 2)
            src_bf_offset_in_dft = k_within_dft % (prev_dft_size // 2)

            # Calculate absolute physical indices of the source butterflies.
            num_bf_per_prev_dft = prev_dft_size // 2
            src_bf_idx_E = (
                src_dft_group_E_idx * num_bf_per_prev_dft + src_bf_offset_in_dft
            )
            src_bf_idx_O = (
                src_dft_group_O_idx * num_bf_per_prev_dft + src_bf_offset_in_dft
            )

            # Map target pairs to source pairs.
            # The output of a source butterfly i is two pairs: (XE_i, XO_i).
            # The pair XE_i is at source pair index 2 * i.
            # The pair XO_i is at source pair index 2 * i + 1.
            source_pair_offset = 1 if is_second_half else 0

            source_pair_for_E = 2 * src_bf_idx_E + source_pair_offset
            source_pair_for_O = 2 * src_bf_idx_O + source_pair_offset

            # The input to the target butterfly k_bf_target is a pair of complex numbers (E, O).
            # The target pair for E is at index 2 * k_bf_target.
            # The target pair for O is at index 2 * k_bf_target + 1.
            perm_indices_pairs[2 * k_bf_target] = source_pair_for_E
            perm_indices_pairs[2 * k_bf_target + 1] = source_pair_for_O

        return perm_indices_pairs

    def _create_final_unscramble_indices(self) -> np.ndarray:
        """
        Creates permutation indices for complex pairs to unscramble the output
        of the final butterfly stage into natural order.
        Returns a permutation map of size N.
        """
        perm_indices_pairs = np.empty(self.N, dtype=int)
        num_butterflies = self.N // 2

        for k in range(num_butterflies):
            # The output of the final conv layer is a sequence of pairs (X_k, X_{k+N/2}).
            # The source pair for X_k is at index 2*k.
            # The source pair for X_{k+N/2} is at index 2*k+1.

            # We want to move the source pair for X_k to the target position k.
            perm_indices_pairs[k] = 2 * k

            # We want to move the source pair for X_{k+N/2} to the target position k+N/2.
            perm_indices_pairs[k + num_butterflies] = 2 * k + 1

        return perm_indices_pairs

    def _create_butterfly_weights(self, stage_idx: int) -> torch.Tensor:
        """
        Creates the weights for the grouped 1x1 convolution that performs the
        butterfly operations for a given FFT stage.
        """
        # N_b is the size of the DFTs being computed at this stage.
        N_b = 2 ** (stage_idx + 1)
        # Twiddle factors repeat. We only need to calculate N_b/2 unique ones.
        num_unique_butterflies = N_b // 2
        # This pattern of unique butterflies is repeated for each major DFT group.
        num_repetitions = self.N // N_b
        # Get all twiddle factors for a DFT of size N_b.
        dft_mat_Nb = create_dft_matrix(n=N_b)

        in_channels_per_group = 4
        unique_weights = torch.zeros(
            (
                num_unique_butterflies * in_channels_per_group,
                in_channels_per_group,
                1,
                1,
            ),
            dtype=torch.float32,
        )

        for k in range(num_unique_butterflies):
            # Extract twiddle factor W_{N_b}^k from the column for j=1 of the DFT matrix.
            real_Wk = dft_mat_Nb[2 * k, 2].item()
            imag_Wk = dft_mat_Nb[2 * k + 1, 2].item()

            offset = k * in_channels_per_group
            weight_slice = unique_weights[
                offset : offset + in_channels_per_group, :, 0, 0
            ]

            # The butterfly operation is: X_even = E + W_k*O; X_odd = E - W_k*O
            # Inputs: [ReE, ImE, ReO, ImO], Outputs: [Re(XE), Im(XE), Re(XO), Im(XO)]

            # Row for Re(XE) = Re(E) + Re(W*O) = Re(E) + Re(W)*Re(O) - Im(W)*Im(O)
            weight_slice[0, 0] = 1.0
            weight_slice[0, 2] = real_Wk
            weight_slice[0, 3] = -imag_Wk
            # Row for Im(XE) = Im(E) + Im(W*O) = Im(E) + Im(W)*Re(O) + Re(W)*Im(O)
            weight_slice[1, 1] = 1.0
            weight_slice[1, 2] = imag_Wk
            weight_slice[1, 3] = real_Wk
            # Row for Re(XO) = Re(E) - Re(W*O) = Re(E) - Re(W)*Re(O) + Im(W)*Im(O)
            weight_slice[2, 0] = 1.0
            weight_slice[2, 2] = -real_Wk
            weight_slice[2, 3] = imag_Wk
            # Row for Im(XO) = Im(E) - Im(W*O) = Im(E) - Im(W)*Re(O) - Re(W)*Im(O)
            weight_slice[3, 1] = 1.0
            weight_slice[3, 2] = -imag_Wk
            weight_slice[3, 3] = -real_Wk

        # The full weight tensor is created by tiling the unique weights block.
        return unique_weights.tile((num_repetitions, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FFTCNNModule."""
        for layer in self.layers:
            x = layer(x)
        return x
