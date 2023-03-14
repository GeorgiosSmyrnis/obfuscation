# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class definitions for the feature extraction mapping.

This file contains class definitions for the feature extraction and clean to
obfuscated mappings used by our proposed method. The base class EmbeddingMapper
is used as an inferface for the embedding maps that we shall use. The classes
FeatureExtractor and FeatureExtractorWithClassifier make use of these embedding
mappers to create embeddings for the obfuscated images and use them for
classification.
"""

import abc

import clip

import json

from typing import Union, Tuple, Sequence, Optional

import torch


class EmbeddingMapper(torch.nn.Module, metaclass=abc.ABCMeta):
  """Model that maps input embeddings to output embeddings in the same dimensionality.

  This is used as a template for classes implementing a mapping between
  embeddings. Classes that inherit this class must implement an appropriate call
  function.
  """

  def __init__(self):
    super().__init__()
    pass

  @abc.abstractmethod
  def forward(self, inputs: torch.Tensor) -> Union[torch.Tensor, Sequence[torch.Tensor]]:  # pytype: disable=signature-mismatch
    """Abstract call function to be implemented by subclasses.

    Args:
      inputs: the input embeddings that the class must operate on.

    Returns:
      Either a single tensor which corresponds to the output embeddings, or a
      tuple of tensors, one of which is the output embedding and the rest are
      necessary components of the optimization process for each particular
      subclass.
    """


class IdentityEmbeddingMapper(EmbeddingMapper):
  """Placeholder class for an identity mapping.

  This class simply returns the mappings provided to the input. It exists for
  compatibility with the rest of the codebase.
  """

  def __init__(self):
    super().__init__()
    pass

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pytype: disable=signature-mismatch
    return inputs


class MLPEmbeddingMapper(EmbeddingMapper):
  """Mapping from input to output embeddings using an MLP.

  This functions wraps a model that maps an embedding vector to another one in
  the same space (with the intent being to map obfuscated embeddings to clean
  ones). The mapping is performed using multi-layer perceptron.

  Attributes:
    mapping: Mapping MLP, taking obfuscated embeddings as input and
      returning clean ones.
  """

  def __init__(self,
               input_dim: int,
               embed_dim: int,
               mlp_sizes: Sequence[int],
               weight_decay: float = 1e-4,
               final_activation: Optional[str] = 'relu'):
    super().__init__()

    layers = []
    for i in range(len(mlp_sizes)):
      layers.append(torch.nn.Linear(mlp_sizes[i-1] if i > 0 else input_dim, mlp_sizes[i]))
      layers.append(torch.nn.ReLU())

    layers.append(torch.nn.Linear(mlp_sizes[-1], embed_dim))
    if final_activation == "relu":
        layers.append(torch.nn.ReLU())
    elif final_activation == "softmax":
        layers.append(torch.nn.Softmax())

    self.mapping = torch.nn.Sequential(*layers)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pytype: disable=signature-mismatch
    return self.mapping(inputs)


class AutoEncoderEmbeddingMapper(EmbeddingMapper):
  """AutoEncoder style architecture to map between embeddings.

  This class implements an autoencoder style architecture between two
  embedding spaces. This consists of a single encoder and one or more decoder
  heads from the latent dimension to the output. Both the encoder and the
  decoders are implemented as MLPs. Optionally, a skip connection may be added
  from the input embedding space to each of the output embedding spaces.

  When calling this layer, the input is assumed to be a 2-dimensional tensor, of
  shape (batch_size, embed_dim). The output is a 3-dimensional tensor, of
  shape (batch_size, num_decoders, embed_dim) - one extra dimension for the
  varying number of decoder heads.

  Attributes:
    encoder: The MLP mapping from input embedding to latent dimension.
    decoders: A list of decoder MLPs, mapping from latent dimension to the
      various output spaces.
    skip_connection: Whether to add a skip connection from the input embedding
      space to each of the output embedding spaces.
  """

  def __init__(self,
               mlp_sizes: Sequence[int],
               embed_dim: int,
               num_decoders: int = 1,
               weight_decay: float = 1e-4,
               skip_connection: bool = True):
    super().__init__()
    if len(mlp_sizes) % 2 == 0:  
      raise ValueError('In this, mlp_sizes must contain an odd number of'
                       'elements. The middle one corresponds to the latent'
                       'dimension of the autoencoder, and the rest to the sizes'
                       'of the encoder and the decoder (first half and second'
                       'half, respectively).')

    num_layers_encoder = len(mlp_sizes) // 2
    encoder_mlp_sizes = mlp_sizes[:num_layers_encoder]
    latent_dim = mlp_sizes[num_layers_encoder]
    decoder_mlp_sizes = mlp_sizes[num_layers_encoder+1:]

    self.encoder = MLPEmbeddingMapper(
        embed_dim,
        latent_dim,
        encoder_mlp_sizes,
        weight_decay
    )

    self.decoders = []
    for _ in range(num_decoders):
      decoder = MLPEmbeddingMapper(latent_dim, embed_dim, decoder_mlp_sizes, weight_decay)
      self.decoders.append(decoder)

    self.skip_connection = skip_connection

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pytype: disable=signature-mismatch
    """Method to apply the autoencoder based mapping.

    Args:
      inputs: A 2-dimensional tensor, of shape (batch_size, embed_dim).

    Returns:
      result: A 3-dimensional tensor, of shape (batch_size, num_decoders,
        embed_dim)
    """
    x = self.encoder(inputs)
    decoder_outputs = []
    for i in range(len(self.decoders)):
      decoder = self.decoders[i]
      out = decoder(x)
      if self.skip_connection:
        out = out + inputs
      out = out.unsqueeze(dim=1)
      decoder_outputs.append(out)

    result = torch.cat(decoder_outputs, dim=1)
    return result


class VAEEmbeddingMapper(EmbeddingMapper):
  """Embeddings mapping based on a Variational AutoEncoder (VAE).

  This class generates clean embeddings from the obfuscated ones using a VAE.
  More specifically, the encoder architecture generates a mean and a log
  variance for the latent normal distribution, the components of which are
  uncorrelated. These are then used to generate samples for the decoder.

  Both the encoder and decoder architectures are based on MLPs.

  Attributes:
    encoder: VAE encoder, defined as an MLP.
    decoder: VAE decoder, defined as an MLP.
    encoder_mean: Layer encoding the mean of the latent normal distribution.
    encoder_logvar: Layer encoding the log variance of the latent normal
      distribution.
  """

  def __init__(self,
               mlp_sizes: Sequence[int],
               embed_dim: int,
               weight_decay: float = 1e-4):
    super().__init__()

    if len(mlp_sizes) % 2 == 0:
      raise ValueError('In the case of VAE, mlp_sizes must contain an odd'
                       'number of elements. The middle one corresponds to the'
                       'latent dimension of the VAE, and the rest to the sizes'
                       'of the encoder and the decoder (first half and second'
                       'half, respectively).')

    num_layers_encoder = len(mlp_sizes) // 2
    encoder_mlp_sizes = mlp_sizes[:num_layers_encoder]
    latent_dim = mlp_sizes[num_layers_encoder]
    decoder_mlp_sizes = mlp_sizes[num_layers_encoder+1:]

    self.encoder = MLPEmbeddingMapper(embed_dim, latent_dim, encoder_mlp_sizes,
                                      weight_decay)
    self.decoder = MLPEmbeddingMapper(latent_dim, embed_dim, decoder_mlp_sizes,
                                      weight_decay)

    self.encoder_mean = torch.nn.Linear(latent_dim, latent_dim)

    self.encoder_logvar = torch.nn.Linear(latent_dim, latent_dim)

  def forward(self, inputs: torch.Tensor,  # pytype: disable=signature-mismatch
           training: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = self.encoder(inputs)
    z_mean = self.encoder_mean(x)
    z_log_var = self.encoder_logvar(x)

    # During training, generate samples normally.
    if training:
      sample = torch.randn_like(z_mean)
    # During testing, use the means of the distribution to generate
    # representations.
    else:
      sample = torch.zeros_like(z_mean)
    y = z_mean + torch.exp(z_log_var) * sample
    y = self.decoder(y)
    return y, z_mean, z_log_var


class ParameterGenerationEmbeddingMapper(EmbeddingMapper):
  """Class that implements an autoencoder that uses a context dependent decoder.

  This class implements an autoencoder architecture that makes use of a set of
  parameter generators, that generate the parameters of the decoder to be used.
  These generators are context dependent - they are provided with context which
  is learned by another part of the model. This class makes use of a separate
  generator for each layer of the decoder.

  All separate parts of this architecture are implemented as MLPs. The decoder
  architecture is assumed to be symmetric to the encoder. The parameter
  generators all have the same architecture.

  Attributes:
    encoder: The common encoder model mapping from embedding space to latent
      space.
    context: The context oracle that derives the context vector from each of the
      provided embeddings. This means that the model predicts (is an oracle of)
      the context of the input, which correspons to the obfuscation type.
    param_dims: The dimensions of the intermediate vectors of the decoder, to
      which the parameter generators must adhere. This corresponds to the
      architecture of the decoders as MLPs.
    param_generator_list: The list of parameter generators for this model.
  """

  def __init__(
      self,
      encoder_decoder_mlp_sizes: Sequence[int],
      param_generator_mlp_sizes: Sequence[int],
      context_mlp_sizes: Sequence[int],
      embed_dim: int,
      latent_dim: int,
      context_dim: int,
      num_contexts: int = 0,
      weight_decay: float = 1e-4,
  ):
    """Init function.

    Args:
      encoder_decoder_mlp_sizes: The layer sizes of the encoder and the decoder
        architectures of the model.
      param_generator_mlp_sizes: The layer sizes of the parameter generator
        architecture.
      context_mlp_sizes: The layer sizes of the context oracle.
      embed_dim: The dimension of the embedding vectors.
      latent_dim: The latent dimension of the diffusion model.
      context_dim: The dimension of the context vectors.
      num_contexts: How many different domains the model should generate. If
        greater than 0, makes the model generate embeddings.
      weight_decay: L2 weight decay to add to the parameters of the model.
        Defaults to L2.
    """
    super().__init__()

    self.encoder = MLPEmbeddingMapper(
        embed_dim,
        latent_dim,
        encoder_decoder_mlp_sizes,
        weight_decay
    )

    self.context = MLPEmbeddingMapper(
        1,
        context_dim,
        context_mlp_sizes,
        weight_decay
    )

    decoder_mlp_sizes = encoder_decoder_mlp_sizes[::-1]
    self.num_contexts = num_contexts
    self.generation = num_contexts > 0
    self.param_dims = [latent_dim] + list(decoder_mlp_sizes) + [embed_dim]
    self.param_generator_list = []
    for i in range(len(self.param_dims)-1):
      # The generated parameters are param_dims[i] * param_dims[i+1] for the
      # weight matrix, plus param_dims[i+1] for the bias.
      param_generator_output_dim = (self.param_dims[i]+1)*self.param_dims[i+1]

      # TODO(smyrnisg): Make this a single Dense layer.
      param_generator = MLPEmbeddingMapper(context_dim,
                                           param_generator_output_dim,
                                           param_generator_mlp_sizes,
                                           weight_decay).cuda()
      self.param_generator_list.append(param_generator)

  def forward(self, inputs: torch.Tensor, contexts: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:  # pytype: disable=signature-mismatch
    """Derive the obfuscation context and apply the generated decoder.

    This method returns both the generated embeddings and the learned context,
    in order to optimize the context oracle. The context is trained so as to be
    representative of the obfuscation type of the image, in order to be given as
    input to the parameter generator afterwards, for the latter to give us the
    parameters of the correct decoder.

    Args:
      inputs: The input embeddings given to the model.

    Returns:
      A tuple containing the generated embeddings and the derived context
        vector.
    """
    latent_vec = self.encoder(inputs)
    batch_size = latent_vec.shape[0]
    if self.generation:
      context_vec = contexts
      context_vec = self.context(context_vec)
      result = latent_vec
    else:
      raise NotImplementedError()
    for i in range(len(self.param_generator_list)):
      params = self.param_generator_list[i](context_vec)
      params = torch.reshape(
          params, [-1, self.param_dims[i]+1, self.param_dims[i+1]]
      )
      result = torch.matmul(result, params[:, :-1, :]) + params[:, -1:, :]

    result = result[:, 0, :]  # Remove the extra axis.

    return result, context_vec


class DiffusionEmbeddingMapper(EmbeddingMapper):
  """Diffusion model mapping embeddings from one domain to the other.

  This class implements a generator according to the techniques proposed in
  https://arxiv.org/pdf/2006.11239.pdf. In particular, this class implements a
  diffusion process in order to generate samples which attempt to mimic the
  images of the domain it was trained once

  During training, this class outputs a prediction of the noise added to the
  image, as well as the noise itself.

  In order to generate samples, a point from the normal latent space is sampled,
  and the reverse diffusion process is iteratively solved, in order to arrive at
  the generated image (without extra noise).

  In this class, betas, alphas and alphas_bar are defined as in the paper (see
  https://arxiv.org/pdf/2006.11239.pdf for more details).

  Attributes:
    encoder: The encoder part of the architecture.
    decoder: The decoder part of the architecture. Note that this also receives
      a timestep as input, in order to predict the noise at a particular
      timestep.
    concat_layer: Concatenation layer between the encoder and decoder.
    total_time: The total number of timesteps to run the diffusion process.
    betas: The values of beta used in the diffusion.
    alphas: The values of alpha used in the diffusion.
    alphas_bar: The values of alpha_bar used in the diffusion.
    num_points: Number of points in time to pick during training.
  """

  def __init__(self,
               mlp_sizes: Sequence[int],
               embed_dim: int,
               total_time: int = 100,
               weight_decay: float = 1e-4,
               num_points: int = 1):
    super().__init__()

    if len(mlp_sizes) % 2 == 0:
      raise ValueError(
          'In this case, mlp_sizes must be a list of odd length. The first half'
          'of the list corresponds to the encoder part of the diffusion process'
          'while the second part corresponds to the decoder part. The middle'
          'element corresponds to the latent dimension.'
      )

    num_layers_encoder = len(mlp_sizes) // 2
    encoder_mlp_sizes = mlp_sizes[:num_layers_encoder]
    latent_dim = mlp_sizes[num_layers_encoder]
    decoder_mlp_sizes = mlp_sizes[num_layers_encoder+1:]

    self.encoder = MLPEmbeddingMapper(
        embed_dim,
        latent_dim,
        encoder_mlp_sizes,
        weight_decay,
        final_activation=None
    )

    self.decoder = MLPEmbeddingMapper(
        latent_dim+1,
        embed_dim,
        decoder_mlp_sizes,
        weight_decay,
        final_activation=None
    )

    self.total_time = total_time

    # Below definitions are used as in https://arxiv.org/pdf/2006.11239.pdf.
    self.betas = torch.linspace(1e-4, 2e-2, self.total_time)
    self.alphas = 1 - self.betas
    self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    if torch.cuda.is_available():
        self.betas = self.betas.cuda()
        self.alphas = self.alphas.cuda()
        self.alphas_bar = self.alphas_bar.cuda()

    self.num_points = num_points

  def _noise_prediction(self, inputs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Predict the noise to add to the input at a given timestep.

    Args:
      inputs: The input to add noise to.
      t: The tensor containing the timesteps for the predictions of this batch.

    Returns:
      The noise prediction for the input at the given timestep.
    """
    x = self.encoder(inputs)
    x = torch.cat([x, t], dim=-1)
    x = self.decoder(x)
    return x

  def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # pytype: disable=signature-mismatch

    multiple_inputs = torch.tile(inputs, (self.num_points, 1))

    t = torch.randint_like(multiple_inputs[..., [-1]], low=0, high=self.total_time, dtype=torch.int64)
    noise = torch.randn_like(multiple_inputs)

    chosen_alphas_bar = torch.gather(self.alphas_bar, 0, t.view(-1))

    noisy_image_1 = multiple_inputs * torch.sqrt(chosen_alphas_bar).unsqueeze(1)
    
    noisy_image_2 = noise * torch.sqrt(1 - chosen_alphas_bar).unsqueeze(1)
    
    noisy_image = noisy_image_1 + noisy_image_2

    return self._noise_prediction(noisy_image, t.float() / self.total_time), noise


  def get_sample(self, embedding_prior: torch.Tensor, training: bool = False) -> torch.Tensor:
    if training:
      return self._get_sample_train(embedding_prior)
    else:
      return self._get_sample_eval(embedding_prior)

  def _get_sample_train(self, embedding_prior: torch.Tensor) -> torch.Tensor:
    result = torch.randn_like(embedding_prior) + embedding_prior
    for i in range(self.total_time-1, -1, -1):
        z = torch.randn_like(result) if i > 1 else torch.zeros_like(result)
        sigma = torch.sqrt(self.betas[i])
        model_factor = (1 - self.alphas[i])/torch.sqrt(1-self.alphas_bar[i])
        result = result - model_factor * self._noise_prediction(
            result, torch.full_like(result[..., [-1]], i).float() / self.total_time
        )
        result = result / torch.sqrt(self.alphas[i]) + sigma * z

    return result


  def _get_sample_eval(self, embedding_prior: torch.Tensor) -> torch.Tensor:
    """Return a batch of samples from the diffusion process.

    Args:
      embedding_prior: Embeddings on which to condition generation.

    Returns:
      A batch of samples from the diffusion process.
    """
    with torch.no_grad():
        result = torch.randn_like(embedding_prior) + embedding_prior
        for i in range(self.total_time-1, -1, -1):
            z = torch.randn_like(result) if i > 1 else torch.zeros_like(result)
            sigma = torch.sqrt(self.betas[i])
            model_factor = (1 - self.alphas[i])/torch.sqrt(1-self.alphas_bar[i])
            result = result - model_factor * self._noise_prediction(
                result, torch.full_like(result[..., [-1]], i).float() / self.total_time
            )
            result = result / torch.sqrt(self.alphas[i]) + sigma * z

    return result


class TextWrapper(EmbeddingMapper):

  def __init__(self, base_mapper):
    super().__init__()
    self.base_mapper = base_mapper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-L/14", device=device)
    text_prompts = []
    with open("imagenet_labels.json", "r") as f:
      imagenet_text_labels = json.load(f)
    for label in imagenet_text_labels:
      text_prompts.append(f"a photo of a {label}")
    text_tokens = clip.tokenize(text_prompts).to(device)
    for p in clip_model.parameters():
      p.requires_grad = False
    clip_model.eval()
    self.text_embeds = clip_model.encode_text(text_tokens).float()

  def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    image_outputs = self.base_mapper(inputs)
    
    if labels is not None:
      if isinstance(image_outputs, torch.Tensor):
        image_outputs = [image_outputs]
      text_outputs = self.text_embeds[labels, :]
      return *image_outputs, text_outputs
    else:
      return image_outputs
