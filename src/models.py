import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Encoder(nn.Module):
    """Embedding block producing a latent vector from input.

    Parameters
    ----------
    input_size:
        Dimension of input vectors.
    hidden_size:
        Hidden size for encoder layers.
    num_layers:
        Number of layers for the encoder block.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers:int = 3):
        super().__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers


        self._encoder = nn.RNN(input_size=self._input_size,
                               hidden_size=self._hidden_size,
                               num_layers=self._num_layers,
                               batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input and returns a latent vector.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, seq_len, input_size).

        Returns
        -------
            Latent vector with shape (batch_size, seq_len, hidden_size).
        """
        netout, _ = self._encoder(x)
        return netout

class Decoder(nn.Module):
    """Prediction block.

    Parameters
    ----------
    input_size:
        Dimension of input vector.
    hidden_size:
        Hidden size for the RNN.
    output_size:
        Dimension of output vector.
    sigma_x:
        Std when adding noise to state model.
    sigma_y:
        Std when adding noise to observation model.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 sigma_x: float=0.1,
                 sigma_y: float=0.1):

        super().__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._sigma_x = self._preprocess_sigma(sigma_x, self._hidden_size)
        self._sigma_y = self._preprocess_sigma(sigma_y, self._output_size)

        self._rnn = nn.RNNCell(input_size=self._input_size,
                               hidden_size=self._hidden_size)

        self._ff = nn.Linear(in_features=self._hidden_size,
                             out_features=self._output_size)

        self._noise_x = Normal(loc=0,
                         scale=self._sigma_x)
        self._noise_y = Normal(loc=0,
                         scale=self._sigma_y)

    def forward(self, x: torch.Tensor, noise: bool=False) -> torch.Tensor:
        """Propagate input and returns the output vector.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, seq_len, input_size).
        noise:
            Whether to add noise to the observation and state models.

        Returns
        -------
            Tensor with shape (batch_size, seq_len, hidden_size, output_size).
        """
        # RNN forward pass
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        output = torch.empty((batch_size, seq_len, self._hidden_size))
        h_x = torch.randn(batch_size, self._hidden_size)
        for k in range(seq_len):
            h_x = self._rnn(x[:, k, :], h_x)
            if noise:
                h_x += self._noise_x.sample((batch_size,))

            output[:, k, :] = h_x

        # Fully connected pass
        output = self._ff(output)
        if noise:
            output += self._noise_y.sample((batch_size, seq_len))

        return output

    @staticmethod
    def _preprocess_sigma(sigma, size) -> torch.Tensor:
        """Convert sigma into tensor of dimension `size`.

        If sigma is scalar, values are duplicated.
        Parameters
        ----------
        sigma:
            Sigma as `int`, `float`, `torch.Tensor` or `np.ndarray`.
        size
            Size of the output vector.

        Returns
        -------
            Sigma tensor.

        """
        if isinstance(sigma, float) or isinstance(sigma, int):
            return torch.ones(size) * sigma

        sigma = torch.Tensor(sigma)
        assert sigma.shape[0] == size

        return sigma

