import torch

from mldft.ml.models.components.equiformerv2_components.activation import GateActivation
from mldft.ml.models.components.esen_components.so2_layers import SO2_Convolution
from mldft.ml.models.components.esen_components.so3 import CoefficientMapping


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> None:
    assert torch.allclose(actual, expected, atol=atol, rtol=atol)


def test_gate_activation_m_primary_matches_l_primary():
    torch.manual_seed(0)

    mapping = CoefficientMapping(lmax=3, mmax=2)
    gate = GateActivation(
        lmax=3,
        mmax=2,
        num_channels=4,
        mappingReduced=mapping,
    )

    gating_scalars = torch.randn(5, 12)
    input_tensors = torch.randn(5, mapping.res_size, 4)

    output_l_primary = gate(gating_scalars, input_tensors)
    output_m_primary = gate(
        gating_scalars,
        input_tensors.index_select(1, mapping.to_m_index),
        m_primary=True,
    )

    _assert_close(output_l_primary, output_m_primary.index_select(1, mapping.from_m_index))


def test_so2_convolution_output_m_primary_matches_default_and_gradients():
    torch.manual_seed(0)

    mapping = CoefficientMapping(lmax=3, mmax=2)
    conv = SO2_Convolution(
        sphere_channels=6,
        m_output_channels=5,
        lmax=3,
        mmax=2,
        mappingReduced=mapping,
        internal_weights=False,
        edge_channels_list=[7, 9],
        extra_m0_output_channels=4,
    )

    x_default = torch.randn(8, mapping.res_size, 6, requires_grad=True)
    x_edge_default = torch.randn(8, 7, requires_grad=True)
    output_default, extra_default = conv(x_default, x_edge_default)
    output_grad = torch.randn_like(output_default)
    extra_grad = torch.randn_like(extra_default)
    grads_default = torch.autograd.grad(
        (output_default * output_grad).sum() + (extra_default * extra_grad).sum(),
        (x_default, x_edge_default),
    )

    x_layout = x_default.detach().clone().requires_grad_()
    x_edge_layout = x_edge_default.detach().clone().requires_grad_()
    output_m_primary, extra_layout = conv(
        x_layout,
        x_edge_layout,
        output_m_primary=True,
    )
    output_layout = output_m_primary.index_select(1, mapping.from_m_index)
    grads_layout = torch.autograd.grad(
        (output_layout * output_grad).sum() + (extra_layout * extra_grad).sum(),
        (x_layout, x_edge_layout),
    )

    _assert_close(output_layout, output_default, atol=1e-5)
    _assert_close(extra_layout, extra_default, atol=1e-5)
    _assert_close(grads_layout[0], grads_default[0], atol=1e-5)
    _assert_close(grads_layout[1], grads_default[1], atol=1e-5)


def test_so2_convolution_input_m_primary_matches_default_and_gradients():
    torch.manual_seed(0)

    mapping = CoefficientMapping(lmax=3, mmax=2)
    conv = SO2_Convolution(
        sphere_channels=5,
        m_output_channels=4,
        lmax=3,
        mmax=2,
        mappingReduced=mapping,
        internal_weights=True,
        edge_channels_list=None,
        extra_m0_output_channels=None,
    )

    x_default = torch.randn(6, mapping.res_size, 5, requires_grad=True)
    x_edge_default = torch.randn(6, 3, requires_grad=True)
    output_default = conv(x_default, x_edge_default)
    output_grad = torch.randn_like(output_default)
    grads_default = torch.autograd.grad(
        (output_default * output_grad).sum(),
        (x_default, x_edge_default),
        allow_unused=True,
    )

    x_layout = x_default.detach().clone().requires_grad_()
    x_edge_layout = x_edge_default.detach().clone().requires_grad_()
    output_layout = conv(
        x_layout.index_select(1, mapping.to_m_index),
        x_edge_layout,
        input_m_primary=True,
    )
    grads_layout = torch.autograd.grad(
        (output_layout * output_grad).sum(),
        (x_layout, x_edge_layout),
        allow_unused=True,
    )

    _assert_close(output_layout, output_default, atol=1e-5)
    _assert_close(grads_layout[0], grads_default[0], atol=1e-5)
    assert grads_default[1] is None
    assert grads_layout[1] is None


def test_so2_convolution_forward_edge_pair_matches_default_and_gradients():
    torch.manual_seed(0)

    mapping = CoefficientMapping(lmax=3, mmax=2)
    conv = SO2_Convolution(
        sphere_channels=6,
        m_output_channels=5,
        lmax=3,
        mmax=2,
        mappingReduced=mapping,
        internal_weights=False,
        edge_channels_list=[7, 9],
        extra_m0_output_channels=4,
    )

    x_source_default = torch.randn(6, mapping.res_size, 3, requires_grad=True)
    x_target_default = torch.randn(6, mapping.res_size, 3, requires_grad=True)
    x_edge_default = torch.randn(6, 7, requires_grad=True)
    wigner = torch.randn(6, mapping.res_size, mapping.res_size)

    x_message_default = torch.cat((x_source_default, x_target_default), dim=2)
    output_default, extra_default = conv(torch.bmm(wigner, x_message_default), x_edge_default)
    output_grad = torch.randn_like(output_default)
    extra_grad = torch.randn_like(extra_default)
    grads_default = torch.autograd.grad(
        (output_default * output_grad).sum() + (extra_default * extra_grad).sum(),
        (x_source_default, x_target_default, x_edge_default),
    )

    x_source_pair = x_source_default.detach().clone().requires_grad_()
    x_target_pair = x_target_default.detach().clone().requires_grad_()
    x_edge_pair = x_edge_default.detach().clone().requires_grad_()
    output_pair, extra_pair = conv.forward_edge_pair(
        x_source_pair,
        x_target_pair,
        wigner,
        x_edge_pair,
    )
    grads_pair = torch.autograd.grad(
        (output_pair * output_grad).sum() + (extra_pair * extra_grad).sum(),
        (x_source_pair, x_target_pair, x_edge_pair),
    )

    _assert_close(output_pair, output_default, atol=1e-5)
    _assert_close(extra_pair, extra_default, atol=1e-5)
    _assert_close(grads_pair[0], grads_default[0], atol=1e-5)
    _assert_close(grads_pair[1], grads_default[1], atol=1e-5)
    _assert_close(grads_pair[2], grads_default[2], atol=1e-5)


def test_so2_convolution_forward_edge_pair_matches_default_double_backward():
    torch.manual_seed(0)

    mapping = CoefficientMapping(lmax=2, mmax=2)
    conv = SO2_Convolution(
        sphere_channels=4,
        m_output_channels=3,
        lmax=2,
        mmax=2,
        mappingReduced=mapping,
        internal_weights=False,
        edge_channels_list=[5, 7],
        extra_m0_output_channels=2,
    ).double()

    x_source_default = torch.randn(4, mapping.res_size, 2, dtype=torch.double, requires_grad=True)
    x_target_default = torch.randn(4, mapping.res_size, 2, dtype=torch.double, requires_grad=True)
    x_edge_default = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
    wigner = torch.randn(4, mapping.res_size, mapping.res_size, dtype=torch.double)

    x_message_default = torch.cat((x_source_default, x_target_default), dim=2)
    output_default, extra_default = conv(torch.bmm(wigner, x_message_default), x_edge_default)
    loss_default = output_default.square().mean() + extra_default.square().mean()
    grads_default = torch.autograd.grad(
        loss_default,
        (x_source_default, x_target_default, x_edge_default),
        create_graph=True,
    )
    grad_loss_default = sum(grad.square().mean() for grad in grads_default)
    gradgrads_default = torch.autograd.grad(
        grad_loss_default,
        (x_source_default, x_target_default, x_edge_default),
    )

    x_source_pair = x_source_default.detach().clone().requires_grad_()
    x_target_pair = x_target_default.detach().clone().requires_grad_()
    x_edge_pair = x_edge_default.detach().clone().requires_grad_()
    output_pair, extra_pair = conv.forward_edge_pair(
        x_source_pair,
        x_target_pair,
        wigner,
        x_edge_pair,
    )
    loss_pair = output_pair.square().mean() + extra_pair.square().mean()
    grads_pair = torch.autograd.grad(
        loss_pair,
        (x_source_pair, x_target_pair, x_edge_pair),
        create_graph=True,
    )
    grad_loss_pair = sum(grad.square().mean() for grad in grads_pair)
    gradgrads_pair = torch.autograd.grad(
        grad_loss_pair,
        (x_source_pair, x_target_pair, x_edge_pair),
    )

    _assert_close(output_pair, output_default, atol=1e-8)
    _assert_close(extra_pair, extra_default, atol=1e-8)
    _assert_close(grads_pair[0], grads_default[0], atol=1e-8)
    _assert_close(grads_pair[1], grads_default[1], atol=1e-8)
    _assert_close(grads_pair[2], grads_default[2], atol=1e-8)
    _assert_close(gradgrads_pair[0], gradgrads_default[0], atol=1e-7)
    _assert_close(gradgrads_pair[1], gradgrads_default[1], atol=1e-7)
    _assert_close(gradgrads_pair[2], gradgrads_default[2], atol=1e-7)
