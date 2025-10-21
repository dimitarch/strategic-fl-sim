import torch


def compute_gradient_metrics(gradients, aggregate_grad):
    """Compute metrics without creating full stacked tensor."""
    grad_norms = []
    cosine_sims = []

    # Flatten aggregate once
    flat_agg = torch.cat([g.flatten() for g in aggregate_grad])
    agg_norm = torch.norm(flat_agg)

    for grad in gradients:
        # Compute gradient norm
        flat_grad = torch.cat([g.flatten() for g in grad])
        grad_norm = torch.norm(flat_grad)
        grad_norms.append(grad_norm.item())

        # Compute cosine similarity
        dot_product = torch.dot(flat_grad, flat_agg)
        cosine_sim = (dot_product / (grad_norm * agg_norm + 1e-8)).item()
        cosine_sims.append(cosine_sim)

    return grad_norms, cosine_sims


def get_gradient_metrics(gradients, aggregate_grad):
    """Get metrics for gradients (used in train function)."""
    grad_norms, cosine_sims = compute_gradient_metrics(gradients, aggregate_grad)

    return {"grad_norms": grad_norms, "cosine_similarities": cosine_sims}
