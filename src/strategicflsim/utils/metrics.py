import torch


def compute_gradient_metrics(gradients, aggregate_grad):
    """Compute gradient similarity metrics. Assume that all gradients are on the same device."""

    flat_grads = torch.stack(
        [torch.cat([g.flatten() for g in grad]) for grad in gradients]
    )  # [num_clients, total_params]
    flat_agg = torch.cat([g.flatten() for g in aggregate_grad])  # [total_params]

    # Vectorized norms :)
    grad_norms = torch.norm(flat_grads, dim=1)  # [num_clients]
    agg_norm = torch.norm(flat_agg)

    # Vectorized cosine similarities :)
    dot_products = torch.matmul(flat_grads, flat_agg)  # [num_clients]
    cosine_sims = dot_products / (
        grad_norms * agg_norm + 1e-8
    )  # Handle division by zero

    return grad_norms.cpu().numpy().tolist(), cosine_sims.cpu().numpy().tolist()


# def compute_gradient_metrics(gradients, aggregate_grad):
#     """Compute gradient similarity metrics."""
#     # Compute gradient norms
#     grad_norms = []
#     for grad in gradients:
#         norm = torch.sum(torch.stack([torch.sum(g**2) for g in grad])).sqrt()
#         grad_norms.append(norm.detach().cpu().numpy().item())

#     # Compute aggregate norm
#     agg_norm = torch.sum(torch.stack([torch.sum(g**2) for g in aggregate_grad])).sqrt()
#     agg_norm = agg_norm.detach().cpu().numpy().item()

#     # Compute cosine similarities
#     cosine_sims = []
#     for grad in gradients:
#         dot_product = torch.sum(
#             torch.stack([torch.sum(g * ag) for g, ag in zip(grad, aggregate_grad)])
#         )

#         grad_norm = torch.sum(torch.stack([torch.sum(g**2) for g in grad])).sqrt()

#         if grad_norm > 0 and agg_norm > 0:
#             cosine_sim = (
#                 (dot_product / (grad_norm * agg_norm)).detach().cpu().numpy().item()
#             )
#         else:
#             cosine_sim = 0.0
#         cosine_sims.append(cosine_sim)

#     return grad_norms, cosine_sims


def get_gradient_metrics(gradients, aggregate_grad):
    """Get metrics for gradients (used in train function)."""
    grad_norms, cosine_sims = compute_gradient_metrics(gradients, aggregate_grad)

    return {"grad_norms": grad_norms, "cosine_similarities": cosine_sims}
