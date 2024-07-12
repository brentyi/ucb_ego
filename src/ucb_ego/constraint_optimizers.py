import dataclasses
from typing import Literal

import numpy as np
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm
from typing_extensions import assert_never

from . import fncsmpl, fncsmpl_extensions, network
from .transforms import SE3, SO3


def aligned_forward_kinematics(
    shaped_body: fncsmpl.SmplShaped,
    Ts_world_cpf: torch.Tensor,
    T_head_cpf: torch.Tensor,
    body_quats: Tensor,
    left_hand_quats: Tensor | None,
    right_hand_quats: Tensor | None,
    device: torch.device,
):
    batch, timesteps = body_quats.shape[:2]
    posed = shaped_body.with_pose_decomposed(
        T_world_root=SE3.identity(device=device, dtype=body_quats.dtype).wxyz_xyz,
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
    )

    # The root joint is located at the body output world, since we pass in identity for T_world_root.
    T_root_head = posed.Ts_world_joint[:, :, 14, :]
    assert T_root_head.shape == (batch, timesteps, 7)
    Ts_root_joint = posed.Ts_world_joint
    assert Ts_root_joint.shape == (batch, timesteps, 51, 7)

    T_cpf_root = (SE3(T_root_head) @ SE3(T_head_cpf)).inverse().wxyz_xyz
    assert T_cpf_root.shape == (batch, timesteps, 7)

    T_world_root = (SE3(Ts_world_cpf) @ SE3(T_cpf_root)).wxyz_xyz
    assert T_world_root.shape == (batch, timesteps, 7)
    Ts_world_joint = (SE3(T_world_root[:, :, None, :]) @ SE3(Ts_root_joint)).wxyz_xyz
    return Ts_world_joint


def do_constraint_optimization(
    Ts_world_cpf: Float[Tensor, "batch time 7"],
    traj: network.EgoDenoiseTraj,
    body_model: fncsmpl.SmplModel,
    optimizer: Literal["adam", "lbfgs"],
    optimizer_iters: int,
) -> network.EgoDenoiseTraj:
    """Run an optimizer to apply hand detection and foot contact constraints."""
    body_quats = SO3.from_matrix(traj.body_rotmats).wxyz
    hand_quats = (
        SO3.from_matrix(traj.hand_rotmats).wxyz
        if traj.hand_rotmats is not None
        else None
    )
    (batch, timesteps) = body_quats.shape[:2]
    assert body_quats.shape == (batch, timesteps, 21, 4)
    assert traj.betas.shape == (batch, timesteps, 16)
    assert Ts_world_cpf.shape == (batch, timesteps, 7)

    device = body_quats.device
    shaped_body = body_model.with_shape(traj.betas)
    body_deltas = torch.zeros(
        (batch, timesteps, 21, 3), requires_grad=True, device=device
    )

    T_head_cpf = fncsmpl_extensions.get_T_head_cpf(shaped_body)
    assert T_head_cpf.shape == (batch, timesteps, 7)
    initial_Ts_world_joint = aligned_forward_kinematics(
        shaped_body,
        Ts_world_cpf=Ts_world_cpf,
        T_head_cpf=T_head_cpf,
        body_quats=body_quats,
        left_hand_quats=hand_quats[..., :15, :] if hand_quats is not None else None,
        right_hand_quats=hand_quats[..., 15:30, :] if hand_quats is not None else None,
        device=device,
    )

    # Make sure we're not backpropagating into trajectories. Probably unnecessary.
    traj = traj.map(lambda tensor: tensor.detach())

    assert traj.contacts.shape == (batch, timesteps, 21)
    pairwise_contacts = torch.logical_and(
        traj.contacts[:, :-1, :] > 0.5, traj.contacts[:, 1:, :] > 0.5
    )
    assert pairwise_contacts.shape == (batch, timesteps - 1, 21)

    def compute_constraint_loss() -> Tensor:
        perturbed_body_quats = (SO3(body_quats) @ SO3.exp(body_deltas)).wxyz

        Ts_world_joint = aligned_forward_kinematics(
            shaped_body,
            Ts_world_cpf=Ts_world_cpf,
            T_head_cpf=T_head_cpf,
            body_quats=perturbed_body_quats,
            left_hand_quats=None,
            right_hand_quats=None,
            device=device,
        )
        assert Ts_world_joint.shape == (batch, timesteps, 51, 7)

        floor_pad = 0.02
        body_joints_world = Ts_world_joint[..., :21, 4:7]
        on_floor_contact_loss = torch.sum(
            (body_joints_world[:, :, :, 2] * traj.contacts - floor_pad) ** 2 * 0.1
        )
        above_floor_loss = torch.sum(
            torch.minimum(
                body_joints_world[:, :, :, 2], body_joints_world.new_tensor(floor_pad)
            )
            ** 2
        )
        sliding_loss = torch.sum(
            (
                (body_joints_world[:, :-1, :, :] - body_joints_world[:, 1:, :, :])
                * pairwise_contacts[:, :, :, None]
            )
            ** 2
        )
        body_smoothness_loss = torch.sum(
            (body_deltas[:, :-1, :, :] - body_deltas[:, 1:, :, :]) ** 2
        )
        body_delta_reg_loss = torch.sum(body_deltas**2)
        joint_pos_reg_loss = torch.sum(
            (initial_Ts_world_joint[..., :19, 4:7] - Ts_world_joint[..., :19, 4:7]) ** 2
        )
        return (
            5.0 * sliding_loss
            + 1.0 * body_smoothness_loss
            + 0.001 * body_delta_reg_loss
            + 0.001 * joint_pos_reg_loss
            + 0.0 * on_floor_contact_loss
            + 0.1 * above_floor_loss
        )

    if optimizer == "adam":
        opt = torch.optim.Adam([body_deltas], lr=1e-3)
        start_lr = 1e-2
        end_lr = 1e-3
        for i in tqdm(range(optimizer_iters)):
            # Cosine decay for learning rate.
            for param_group in opt.param_groups:
                param_group["lr"] = end_lr + (start_lr - end_lr) * np.cos(
                    i / optimizer_iters * (torch.pi / 2.0)
                )

            opt.zero_grad()
            loss = compute_constraint_loss()
            loss.backward()
            opt.step()

    elif optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            [body_deltas],
            history_size=20,
            max_iter=10,
            tolerance_grad=1e-4,
            tolerance_change=1e-5,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss = compute_constraint_loss()
            loss.backward()
            return loss

        for i in tqdm(range(optimizer_iters)):
            opt.step(closure)  # type: ignore

    else:
        assert_never(optimizer)

    return dataclasses.replace(
        traj,
        body_rotmats=einsum(
            traj.body_rotmats,
            SO3.exp(body_deltas.detach()).as_matrix(),
            "batch time joint i j, batch time joint j k -> batch time joint i k",
        ),
    )
