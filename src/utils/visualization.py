import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def save_calibration_figures(true_normalized, best_solution, best_loss, loss_history, figure_dir):
    """Create and save calibration figures."""
    true_normalized = _to_column_tensor(true_normalized)
    pred_normalized = _to_column_tensor(best_solution)

    true_cumulative = torch.expm1(true_normalized)
    pred_cumulative = torch.expm1(pred_normalized)

    loss_normalized = nn.MSELoss()(pred_normalized, true_normalized).item()
    loss_cumulative = nn.MSELoss()(pred_cumulative, true_cumulative).item()

    # Normalized cases plot
    fig1, ax_fig1 = plt.subplots()
    ax_fig1.plot(true_normalized.squeeze(-1).numpy(), label='True Cases')
    ax_fig1.plot(pred_normalized.squeeze(-1).numpy(), label='Predicted Cases')
    ax_fig1.set_xlabel('Time (weeks)')
    ax_fig1.set_ylabel('Normalized Cases')
    ax_fig1.set_title('Calibration: True vs Predicted Dengue Cases')
    ax_fig1.text(0.5, -0.2, f'Loss (Normalized): {loss_normalized:.4f}',
                 transform=ax_fig1.transAxes, ha='center')
    ax_fig1.legend()
    ax_fig1.grid(True)
    fig1.savefig(
        f'{figure_dir}/calibration_normalized_cases.png',
        bbox_inches='tight',
    )
    plt.close(fig1)

    # Cumulative cases plot
    fig2, ax_fig2 = plt.subplots()
    ax_fig2.plot(true_cumulative.squeeze(-1).numpy(), label='True Cases')
    ax_fig2.plot(pred_cumulative.squeeze(-1).numpy(), label='Predicted Cases')
    ax_fig2.set_xlabel('Time (weeks)')
    ax_fig2.set_ylabel('Cumulative Cases')
    ax_fig2.set_title('Calibration: True vs Predicted Dengue Cases')
    ax_fig2.text(0.5, -0.2, f'Loss (Cumulative): {loss_cumulative:.4f}',
                 transform=ax_fig2.transAxes, ha='center')
    ax_fig2.legend()
    ax_fig2.grid(True)
    fig2.savefig(
        f'{figure_dir}/calibration_cumulative_cases.png',
        bbox_inches='tight',
    )
    plt.close(fig2)

    # Loss history plot
    fig3, ax_fig3 = plt.subplots()
    ax_fig3.plot(loss_history, label='Calibration Loss')
    ax_fig3.set_xlabel('Epoch')
    ax_fig3.set_ylabel('Loss')
    ax_fig3.set_title('Calibration Loss History')
    ax_fig3.text(0.5, -0.2, f'Best Loss: {best_loss:.4f}',
                 transform=ax_fig3.transAxes, ha='center')
    ax_fig3.legend()
    ax_fig3.grid(True)
    fig3.savefig(
        f'{figure_dir}/calibration_loss.png',
        bbox_inches='tight',
    )
    plt.close(fig3)


def save_projection_figures(true_normalized, pred_normalized, figure_dir):
    true_normalized = _to_column_tensor(true_normalized)
    pred_normalized = _to_column_tensor(pred_normalized)

    true_cumulative = torch.expm1(true_normalized)
    pred_cumulative = torch.expm1(pred_normalized)

    loss_normalized = nn.MSELoss()(pred_normalized, true_normalized).item()
    loss_cumulative = nn.MSELoss()(pred_cumulative, true_cumulative).item()

    # Normalized cases plot
    fig1, ax_fig1 = plt.subplots()
    ax_fig1.plot(true_normalized.squeeze(-1).numpy(), label='True Cases')
    ax_fig1.plot(pred_normalized.squeeze(-1).numpy(), label='Predicted Cases')
    ax_fig1.set_xlabel('Time (weeks)')
    ax_fig1.set_ylabel('Normalized Cases')
    ax_fig1.set_title('Projection: True vs Predicted Dengue Cases')
    ax_fig1.text(0.5, -0.2, f'Loss (Normalized): {loss_normalized:.4f}',
                 transform=ax_fig1.transAxes, ha='center')

    ax_fig1.legend()
    ax_fig1.grid(True)
    fig1.savefig(f'{figure_dir}/projection_normalized_cases.png',
                 bbox_inches='tight')
    plt.close(fig1)

    # Cumulative cases plot
    fig2, ax_fig2 = plt.subplots()
    ax_fig2.plot(true_cumulative.squeeze(-1).numpy(), label='True Cases')
    ax_fig2.plot(pred_cumulative.squeeze(-1).numpy(), label='Predicted Cases')
    ax_fig2.set_xlabel('Time (weeks)')
    ax_fig2.set_ylabel('Cumulative Cases')
    ax_fig2.set_title('Projection: True vs Predicted Dengue Cases')
    ax_fig2.text(0.5, -0.2, f'Loss (Cumulative): {loss_cumulative:.4f}',
                 transform=ax_fig2.transAxes, ha='center')
    ax_fig2.legend()
    ax_fig2.grid(True)
    fig2.savefig(f'{figure_dir}/projection_cumulative_cases.png',
                 bbox_inches='tight')
    plt.close(fig2)


def _to_column_tensor(value):
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    if tensor.ndim > 2:
        return tensor.reshape(-1, 1)
    if tensor.shape[-1] != 1:
        return tensor.reshape(-1, 1)
    return tensor
