Model Overview
==============

We attempted to reproduce the architecture as described in [M-OFDFT]_ as accurately as possible.

.. raw:: html
    :file: model_archi_patched.svg


This figure is reproduced from the M-OFDFT preprint, with hyperlinks added to our implementation of the different parts,
whenever there is a one-to-one correspondence to a class in our implementation.

Exceptions are:

* The Coefficient Adapter: We chose to implement the local frames and the natural reparametrization as transforms,
  i.e. as part of the data loading, not the model. On the other hand, the dimension-wise rescaling is implemented
  as the first layer in the model. This ensures that the gradient loss is applied to non-rescaled coefficients, hence
  correctly weighting the different basis functions. While we believe this makes sense, it is not explicitly stated in
  [M-OFDFT]_ and we might implement the other choice in the future (most likely by keeping the structure the same but
  adding weights to the loss function which emulate applying the rescaling outside of the model).

* The MLPs: As no details are given in the paper, we simply use the pytorch implementation.

* The Atom Embedding: It is implemented simply as an instance of :class:`torch.nn.Embedding`, as an attribute of the
  :class:`~mldft.ml.models.components.node_embedding.NodeEmbedding`.

* The layer norms and self attention, which are both handled by the
  :class:`~mldft.ml.models.components.g3d_layer.G3DLayer`.

Additionally to what is show in the figure, we also implement a module for the "projected MINAO" guess, which predicts
the difference of coefficients between the input to the model and the ground state. This is implemented in
:class:`~mldft.ml.models.components.initial_guess_delta_module.InitialGuessDeltaModule`.

Finally, the prediction of the kinetic potential in a variational way is implemented in the
all-encompassing lightning-module for the model, :class:`~mldft.ml.models.mldft_module.MLDFTLitModule`,
in the :meth:`~mldft.ml.models.mldft_module.MLDFTLitModule.forward` method.
