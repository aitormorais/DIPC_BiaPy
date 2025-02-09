import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input

from .tr_layers import TransformerBlock, ClassToken, Patches, PatchEncoder


def ViT(input_shape, patch_size, hidden_size, transformer_layers, num_heads, mlp_head_units, n_classes=1, 
        dropout=0.0, include_class_token=True, representation_size=None, include_top=True, 
        use_as_backbone=False):
    """
    ViT architecture. `ViT paper <https://arxiv.org/abs/2010.11929>`__.

    Parameters
    ----------
    input_shape : 3D/4D tuple
        Dimensions of the input image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.
        
    patch_size : int
        Size of the patches that are extracted from the input image. As an example, to use ``16x16`` 
        patches, set ``patch_size = 16``.

    hidden_size : int
        Dimension of the embedding space.

    transformer_layers : int
        Number of transformer encoder layers.

    num_heads : int
        Number of heads in the multi-head attention layer.

    mlp_head_units : int
        Size of the dense layer of the final classifier. 

    n_classes : int, optional
        Number of classes to predict. Is the number of channels in the output tensor.

    dropout : bool, optional
        Dropout rate for the decoder (can be a list of dropout rates for each layer).

    include_class_token : bool, optional
        Whether to include or not the class token.

    representation_size : int, optional
        The size of the representation prior to the classification layer. If None, no Dense layer is inserted.
        Not used but added to mimic vit-keras. 

    include_top : bool, optional
        Whether to include the final classification layer. If not, the output will have dimensions 
        ``(batch_size, hidden_size)``.

    use_as_backbone : bool, optional
        Whether to use the model as a backbone so its components are returned instead of a composed model.

    Returns
    -------
    model : Keras model, optional
        Model containing the ViT .
        
    inputs : Tensorflow layer, optional
        Input layer.

    hidden_states_out : List of Tensorflow layers, optional 
        Layers of the transformer. 

    encoded_patches : PatchEncoder, optional 
        Patch enconder.
    """
    inputs = layers.Input(shape=input_shape)
    if len(input_shape) == 4:
        dims = 3   
        patch_dims = patch_size*patch_size*patch_size*input_shape[-1]
    else:
        dims = 2
        patch_dims = patch_size*patch_size*input_shape[-1]
    num_patches = (input_shape[0]//patch_size)**dims

    # Patch creation 
    # 2D: (B, num_patches^2, patch_dims)
    # 3D: (B, num_patches^3, patch_dims)
    y = Patches(patch_size, patch_dims, dims)(inputs)

    # Patch encoder
    # 2D: (B, num_patches^2, hidden_size)
    # 3D: (B, num_patches^3, hidden_size)
    y = PatchEncoder(num_patches=num_patches, hidden_size=hidden_size)(y)

    if include_class_token:
        y = ClassToken(name="class_token")(y)
        # 2D: (B, (num_patches^2)+1, hidden_size)
        # 3D: (B, (num_patches^3)+1, hidden_size)

    if use_as_backbone:
        hidden_states_out = []

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        
        # TransformerBlock
        # 2D: (B, num_patches^2, hidden_size)
        # 3D: (B, num_patches^3, hidden_size)
        y, _ = TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_head_units,
            dropout=dropout,
            name=f"Transformer/encoderblock_{i}",
        )(y)
        
        if use_as_backbone:
            hidden_states_out.append(y)

    if use_as_backbone:
        return inputs, hidden_states_out, y

    y = layers.LayerNormalization(epsilon=1e-6)(y)
    if include_class_token:
        y = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    if representation_size is not None:
        y = layers.Dense(hidden_size, name="pre_logits", activation="tanh")(y)
    if include_top:
        y = layers.Dense(n_classes, name="head", activation="linear")(y)
    
    model = Model(inputs=inputs, outputs=y)

    return model
